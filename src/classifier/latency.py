#!/usr/bin/env python
"""Step 4: batch-1 inference latency (tokenization included, model load excluded)."""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from classifier.model import load_classifier, predict_probs  # noqa: E402

MAX_LENGTH = 64


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def load_texts(run_dir: Path, partition: str, n: int, seed: int) -> list[str]:
    path = run_dir / f"eval_{partition}" / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run evaluate.py --partition {partition} first")
    with path.open(encoding="utf-8") as f:
        texts = [json.loads(line)["utt"] for line in f if line.strip()]
    random.Random(seed).shuffle(texts)
    return texts[:n]


def bench(run_dir: Path, device_name: str, texts: list[str], warmup: int) -> dict:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    model, tokenizer, _ = load_classifier(run_dir / "model", device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def one(text: str) -> None:
        predict_probs(model, tokenizer, [text], device, MAX_LENGTH, 1)

    for t in texts[:warmup]:
        one(t)
    lat = []
    for t in texts[warmup:]:
        sync()
        t0 = time.perf_counter()
        one(t)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    a = np.array(lat)
    return {
        "run": run_dir.name, "device": device_name, "n_timed": len(lat), "warmup": warmup,
        "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "n_params": sum(p.numel() for p in model.parameters()),
        "model_dir_mb": dir_size_mb(run_dir / "model"),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[ROOT / "results/runs/e5-small-e15", ROOT / "results/runs/xlmr-base-e10"])
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--partition", default="dev", choices=["dev"])  # timing inputs only, never test text
    ap.add_argument("--n", type=int, default=250, help="timed requests after warmup")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/comparison")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        texts = load_texts(run, args.partition, args.n + args.warmup, args.seed)
        for dev in args.devices:
            r = bench(run, dev, texts, args.warmup)
            results.append(r)
            print(f"{r['run']:<14} {dev:<5} p50={r['p50_ms']:.1f} ms  p95={r['p95_ms']:.1f} ms  "
                  f"mean={r['mean_ms']:.1f} ms  params={r['n_params']:,}  size={r['model_dir_mb']:.0f} MB")
    out = args.out_dir / "latency.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()