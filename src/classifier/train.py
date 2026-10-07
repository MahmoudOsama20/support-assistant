#!/usr/bin/env python
"""Fine-tune a multilingual encoder for MASSIVE intent classification.

Trains on the combined ar-SA + en-US train partitions, picks the best epoch by
dev macro-F1, and never reads the test partition (see evaluate.py).

Usage:
    python src/classifier/train.py --config configs/classifier.yaml
    python src/classifier/train.py --config configs/classifier.yaml --run-name smoke \
        --max-train-samples 256 --max-eval-samples 128 --epochs 1 --batch-size 8 --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.dataset import (  # noqa: E402
    LOCALES,
    Record,
    build_labels,
    check_labels_cover,
    load_massive,
    save_labels,
    select_split,
)
from classifier.metrics import compute_metrics, group_metrics  # noqa: E402
from classifier.model import build_model, resolve_device  # noqa: E402

REQUIRED_KEYS = (
    "model_name", "run_name", "output_dir", "max_length", "batch_size",
    "eval_batch_size", "learning_rate", "weight_decay", "warmup_ratio",
    "epochs", "patience", "seed", "fp16",
)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = {
        "run_name": args.run_name, "model_name": args.model_name,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seed": args.seed,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config {path} is missing keys: {missing}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subsample(records: list[Record], n: int | None, seed: int) -> list[Record]:
    if n is None or n >= len(records):
        return records
    return random.Random(seed).sample(records, n)


def make_collate(tokenizer, label2id: dict[str, int], max_length: int):
    def collate(batch: list[Record]):
        enc = tokenizer(
            [r["utt"] for r in batch],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        enc["labels"] = torch.tensor([label2id[r["intent"]] for r in batch])
        return enc

    return collate


def amp_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def evaluate_split(model, loader, device, use_amp) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss, n = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_context(use_amp):
            out = model(**batch)
        size = batch["labels"].size(0)
        total_loss += out.loss.item() * size
        n += size
        y_true += batch["labels"].tolist()
        y_pred += out.logits.argmax(dim=-1).tolist()
    return total_loss / n, y_true, y_pred


def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def save_checkpoint(model, tokenizer, labels: list[str], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_labels(labels, model_dir / "labels.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "classifier.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-train-samples", type=int, help="smoke tests only")
    parser.add_argument("--max-eval-samples", type=int, help="smoke tests only")
    args = parser.parse_args()

    cfg = load_config(args.config, args)
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    use_amp = bool(cfg["fp16"]) and device.type == "cuda"
    run_dir = PROJECT_ROOT / cfg["output_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {cfg['run_name']} | model: {cfg['model_name']} | device: {device} | fp16: {use_amp}")

    # ---- data (official partitions only; test is never loaded into training) ----
    records = load_massive()
    labels = build_labels(records)
    label2id = {label: i for i, label in enumerate(labels)}
    train_full = select_split(records, "train")
    dev_full = select_split(records, "dev")
    check_labels_cover(train_full + dev_full, labels)
    train = maybe_subsample(train_full, args.max_train_samples, cfg["seed"])
    dev = maybe_subsample(dev_full, args.max_eval_samples, cfg["seed"])
    print(f"train: {len(train)} | dev: {len(dev)} | labels: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    for locale in LOCALES:  # empirical check of max_length
        lengths = sorted(
            len(tokenizer(r["utt"], truncation=False)["input_ids"])
            for r in train_full if r["locale"] == locale
        )
        print(f"  tokens/utterance {locale}: mean={np.mean(lengths):.1f} "
              f"p99={lengths[int(0.99 * len(lengths))]} max={lengths[-1]} "
              f"(max_length={cfg['max_length']})")

    collate = make_collate(tokenizer, label2id, cfg["max_length"])
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate, generator=generator)
    dev_loader = DataLoader(dev, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=collate)

    # ---- model / optimisation ----
    model = build_model(cfg["model_name"], labels).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = make_optimizer(model, cfg["learning_rate"], cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ---- training ----
    history: list[dict[str, Any]] = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    dev_locales = [r["locale"] for r in dev]
    started = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
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
        overall = compute_metrics(y_true, y_pred)
        per_locale = group_metrics(dev_locales, y_true, y_pred)
        entry = {
            "epoch": epoch,
            "train_loss": running / steps,
            "dev_loss": dev_loss,
            "dev_accuracy": overall["accuracy"],
            "dev_macro_f1": overall["macro_f1"],
            "dev_per_locale": per_locale,
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(entry)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        locale_str = " | ".join(f"{loc} F1 {m['macro_f1']:.4f}" for loc, m in per_locale.items())
        print(f"epoch {epoch}: train_loss {entry['train_loss']:.4f} | dev_loss {dev_loss:.4f} | "
              f"acc {overall['accuracy']:.4f} | macro-F1 {overall['macro_f1']:.4f} | "
              f"{locale_str} | {entry['seconds']}s")

        if overall["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = overall["macro_f1"], epoch, 0
            save_checkpoint(model, tokenizer, labels, run_dir / "model")
            print(f"  saved best checkpoint (epoch {epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping: no dev macro-F1 improvement for {cfg['patience']} epochs")
                break

    summary = {
        "run_name": cfg["run_name"],
        "model_name": cfg["model_name"],
        "num_parameters": num_params,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "train_seconds": round(time.time() - started, 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "subsampled": bool(args.max_train_samples or args.max_eval_samples),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Best epoch {best_epoch}, dev macro-F1 {best_f1:.4f}. Model: {run_dir / 'model'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())