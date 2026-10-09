#!/usr/bin/env python
"""Evaluate the trained route model on dev or heldout, using the temperature fit on dev.

Heldout is run ONCE per frozen model: an existing heldout report is not overwritten without --force.

Usage:
    python src/route/evaluate_route.py --split dev
    python src/route/evaluate_route.py --split heldout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.model import load_classifier, resolve_device  # noqa: E402
from route.train_route import ROUTE_LABELS, collect_logits, load_route_rows, route_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "results" / "runs" / "route-e5small")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "route")
    ap.add_argument("--split", required=True, choices=["dev", "heldout"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = args.run_dir / f"route_{args.split}_report.json"
    if args.split == "heldout" and out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists: heldout is evaluated once per frozen model (use --force to override)")

    device = resolve_device(args.device)
    model, tokenizer, labels = load_classifier(args.run_dir / "model", device)
    if list(labels) != ROUTE_LABELS:
        raise SystemExit(f"Model labels {labels} != {ROUTE_LABELS}")
    temperature = json.loads((args.run_dir / "model" / "temperature.json").read_text(encoding="utf-8"))["temperature"]

    rows = load_route_rows(args.data_dir / f"route_{args.split}.jsonl", args.split)
    logits = collect_logits(model, tokenizer, [r["utt"] for r in rows], device, args.max_length, 128)
    report = route_report(logits, rows, temperature)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.split}: n={report['n']} acc {report['accuracy']:.4f} macro-F1 {report['macro_f1']:.4f}")
    for lang, m in report["per_language"].items():
        print(f"  {lang}: n={m['n']} acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f}")
    print(f"ECE raw {report['ece_raw']:.4f} -> scaled {report['ece_scaled']:.4f} (T={temperature:.3f})")
    print(f"Confusion (rows=true, cols=pred, order={ROUTE_LABELS}):")
    for row in report["confusion_rows_true_cols_pred"]:
        print("  ", row)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())