#!/usr/bin/env python
"""Fine-tune the routing classifier (4 classes) from the Part A weights.

Selects the best epoch on route DEV macro-F1, then fits a temperature on DEV.
Never reads the heldout split (see evaluate_route.py).

Usage:
    python src/route/train_route.py --device cuda
    python src/route/train_route.py --run-name smoke --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.metrics import compute_metrics  # noqa: E402
from classifier.model import load_classifier, resolve_device  # noqa: E402
from classifier.train import (  # noqa: E402
    amp_context,
    evaluate_split,
    make_collate,
    make_optimizer,
    save_checkpoint,
    set_seed,
)

ROUTE_LABELS = ["kb_question", "data_lookup", "out_of_scope", "unsafe_request"]
REQUIRED_KEYS = (
    "model_name", "run_name", "output_dir", "data_dir", "max_length", "batch_size",
    "eval_batch_size", "learning_rate", "weight_decay", "warmup_ratio",
    "epochs", "patience", "seed", "fp16",
)


# ---------------------------------------------------------------- data ----
def load_route_rows(path: Path, split: str) -> list[dict[str, str]]:
    """Read route_<split>.jsonl into the dict shape train.py's collate expects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing route data: {path} (run src/route/generate.py)")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"] not in ROUTE_LABELS:
            raise ValueError(f"{path}: unknown label {r['label']!r} in {r.get('id')}")
        if r["split"] != split:
            raise ValueError(f"{path}: row {r.get('id')} has split {r['split']!r}, expected {split!r}")
        rows.append({"id": r["id"], "utt": r["text"], "intent": r["label"], "locale": r["language"]})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def resolve_model_source(name: str) -> str:
    """A repo-relative directory wins; otherwise treat it as a HF model id."""
    local = PROJECT_ROOT / name
    return str(local) if local.is_dir() else name


# ------------------------------------------------------------- metrics ----
def per_language(languages: list[str], y_true: list[int], y_pred: list[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lang in sorted(set(languages)):
        idx = [i for i, l in enumerate(languages) if l == lang]
        m = compute_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx])
        out[lang] = {"n": len(idx), "accuracy": float(m["accuracy"]), "macro_f1": float(m["macro_f1"])}
    return out


@torch.inference_mode()
def collect_logits(model, tokenizer, texts: list[str], device, max_length: int, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        chunks.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax_np(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the max-probability confidence."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid search T in [0.25, 10] minimising NLL (fit on DEV only)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 200))
    return float(grid[int(np.argmin([nll(logits, y, t) for t in grid]))])


def route_report(logits: np.ndarray, rows: list[dict[str, str]], temperature: float) -> dict[str, Any]:
    label2id = {l: i for i, l in enumerate(ROUTE_LABELS)}
    y = np.array([label2id[r["intent"]] for r in rows])
    pred = logits.argmax(axis=1)  # temperature does not change the argmax
    overall = compute_metrics(y.tolist(), pred.tolist())
    langs = [r["locale"] for r in rows]
    return {
        "n": len(rows),
        "labels": ROUTE_LABELS,
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "per_language": per_language(langs, y.tolist(), pred.tolist()),
        "confusion_rows_true_cols_pred": confusion_matrix(y, pred, labels=range(len(ROUTE_LABELS))).tolist(),
        "temperature": temperature,
        "nll_raw": nll(logits, y), "nll_scaled": nll(logits, y, temperature),
        "ece_raw": ece(softmax_np(logits), y), "ece_scaled": ece(softmax_np(logits, temperature), y),
    }


# ---------------------------------------------------------------- main ----
def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    overrides = {"run_name": args.run_name, "model_name": args.init_from, "epochs": args.epochs,
                 "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config {path} is missing keys: {missing}")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "route.yaml")
    ap.add_argument("--run-name")
    ap.add_argument("--init-from", help="model dir or HF id (default: config model_name)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_config(args.config, args)
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    use_amp = bool(cfg["fp16"]) and device.type == "cuda"
    run_dir = PROJECT_ROOT / cfg["output_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_model_source(cfg["model_name"])
    print(f"Run: {cfg['run_name']} | init: {source} | device: {device} | fp16: {use_amp}")

    data_dir = PROJECT_ROOT / cfg["data_dir"]
    train = load_route_rows(data_dir / "route_train.jsonl", "train")
    dev = load_route_rows(data_dir / "route_dev.jsonl", "dev")
    print(f"train: {len(train)} | dev: {len(dev)} | labels: {ROUTE_LABELS}")

    tokenizer = AutoTokenizer.from_pretrained(source)
    label2id = {l: i for i, l in enumerate(ROUTE_LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(ROUTE_LABELS),
        id2label=dict(enumerate(ROUTE_LABELS)), label2id=label2id,
        ignore_mismatched_sizes=True,  # Part A head is 60-way; the 4-way head is new
    ).to(device)

    collate = make_collate(tokenizer, label2id, cfg["max_length"])
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=gen)
    dev_loader = DataLoader(dev, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=collate)

    optimizer = make_optimizer(model, cfg["learning_rate"], cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_f1, best_epoch, bad = -1.0, 0, 0
    dev_langs = [r["locale"] for r in dev]
    started = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, steps = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_context(use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            steps += 1

        dev_loss, y_true, y_pred = evaluate_split(model, dev_loader, device, use_amp)
        m = compute_metrics(y_true, y_pred)
        langs = per_language(dev_langs, y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": running / steps, "dev_loss": dev_loss,
                        "dev_accuracy": float(m["accuracy"]), "dev_macro_f1": float(m["macro_f1"]),
                        "dev_per_language": langs})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        lang_str = " | ".join(f"{k} F1 {v['macro_f1']:.4f}" for k, v in langs.items())
        print(f"epoch {epoch}: train_loss {running / steps:.4f} | dev_loss {dev_loss:.4f} | "
              f"acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | {lang_str}")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = float(m["macro_f1"]), epoch, 0
            save_checkpoint(model, tokenizer, ROUTE_LABELS, run_dir / "model")
            print(f"  saved best checkpoint (epoch {epoch})")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stopping after {cfg['patience']} epochs without dev improvement")
                break

    # ---- temperature scaling on DEV with the best checkpoint ----
    best_model, best_tok, labels = load_classifier(run_dir / "model", device)
    if list(labels) != ROUTE_LABELS:
        raise RuntimeError(f"Saved labels {labels} != {ROUTE_LABELS}")
    logits = collect_logits(best_model, best_tok, [r["utt"] for r in dev], device,
                            cfg["max_length"], cfg["eval_batch_size"])
    y_dev = np.array([label2id[r["intent"]] for r in dev])
    temperature = fit_temperature(logits, y_dev)
    (run_dir / "model" / "temperature.json").write_text(
        json.dumps({"temperature": temperature, "fit_split": "dev", "n": len(dev)}, indent=2), encoding="utf-8")
    report = route_report(logits, dev, temperature)
    (run_dir / "route_dev_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "train_summary.json").write_text(json.dumps({
        "run_name": cfg["run_name"], "init": source, "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1, "temperature": temperature,
        "train_seconds": round(time.time() - started, 1), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }, indent=2), encoding="utf-8")

    print(f"Done. Best epoch {best_epoch}, dev macro-F1 {best_f1:.4f}, T={temperature:.3f}")
    print(f"dev ECE raw {report['ece_raw']:.4f} -> scaled {report['ece_scaled']:.4f} | "
          f"NLL raw {report['nll_raw']:.4f} -> scaled {report['nll_scaled']:.4f}")
    print(f"Confusion (rows=true, cols=pred, order={ROUTE_LABELS}):")
    for row in report["confusion_rows_true_cols_pred"]:
        print("  ", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())