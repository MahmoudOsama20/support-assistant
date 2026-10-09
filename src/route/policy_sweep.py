#!/usr/bin/env python
"""Sweep tau_route on route DEV only (never heldout) using the real decide().

Usage:
    python src/route/policy_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.policy import threshold_sweep  # noqa: E402
from classifier.model import load_classifier, resolve_device  # noqa: E402
from route.train_route import ROUTE_LABELS, collect_logits, load_route_rows, softmax_np  # noqa: E402

GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "results" / "runs" / "route-e5small")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "route")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, tokenizer, labels = load_classifier(args.run_dir / "model", device)
    if list(labels) != ROUTE_LABELS or model.config.num_labels != 4:
        raise SystemExit(f"Unexpected route model: labels={labels}, num_labels={model.config.num_labels}")
    temperature = json.loads((args.run_dir / "model" / "temperature.json").read_text(encoding="utf-8"))["temperature"]

    rows = load_route_rows(args.data_dir / "route_dev.jsonl", "dev")
    logits = collect_logits(model, tokenizer, [r["utt"] for r in rows], device, 64, 128)
    probs = softmax_np(logits, temperature)
    table = threshold_sweep(probs, [r["intent"] for r in rows], list(labels), GRID)

    print(f"route DEV n={len(rows)} T={temperature:.3f} (tool-bound rows = kb_question + data_lookup)")
    head = f"{'tau':>5} {'clarif':>6} {'acc_acc':>7} {'ovr_clar':>8} {'clar_rt':>7} {'misrt':>5} {'ovr_ref':>7} {'uns_leak':>8} {'oos_leak':>8}"
    print(head)
    for r in table:
        print(f"{r['tau']:>5.2f} {r['clarified']:>6} {r['accepted_accuracy']:>7.4f} {r['over_clarify']:>8} "
              f"{r['over_clarify_rate']:>7.3f} {r['tool_misroute']:>5} {r['over_refusal']:>7} "
              f"{r['unsafe_leak']:>8} {r['oos_leak']:>8}")
    out = args.run_dir / "policy_sweep_dev.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())