#!/usr/bin/env python
"""Evaluate a trained classifier on an official partition (default: test).

Reports overall and per-locale accuracy / macro-F1 with bootstrap CIs, saves
predictions, per-locale confusion matrices, per-intent reports and top
confused pairs. Run on test only for final candidates.

Usage:
    python src/classifier/evaluate.py --run-dir results/runs/xlmr-base --partition dev
    python src/classifier/evaluate.py --run-dir results/runs/xlmr-base --partition test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.dataset import (  # noqa: E402
    LOCALES,
    PARTITIONS,
    check_labels_cover,
    load_massive,
    select_split,
)
from classifier.metrics import (  # noqa: E402
    bootstrap_ci,
    build_confusion,
    compute_metrics,
    top_confused_pairs,
    within_scenario_error_share,
)
from classifier.model import load_classifier, predict_probs, resolve_device  # noqa: E402


def write_confusion_csv(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, cm):
            writer.writerow([label, *row.tolist()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partition", default="test", choices=PARTITIONS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000, help="0 disables CIs")
    parser.add_argument("--max-samples", type=int, help="smoke tests only")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    cfg_path = run_dir / "config_resolved.json"
    max_length = json.loads(cfg_path.read_text())["max_length"] if cfg_path.is_file() else 64

    device = resolve_device(args.device)
    model, tokenizer, labels = load_classifier(run_dir / "model", device)
    label2id = {label: i for i, label in enumerate(labels)}

    records = load_massive()
    intent_to_scenario = {r["intent"]: r["scenario"] for recs in records.values() for r in recs}
    split = select_split(records, args.partition)
    check_labels_cover(split, labels)
    if args.max_samples and args.max_samples < len(split):
        split = random.Random(42).sample(split, args.max_samples)

    probs = predict_probs(model, tokenizer, [r["utt"] for r in split], device,
                          max_length=max_length, batch_size=args.batch_size)
    confidences, preds = probs.max(dim=-1)
    y_pred = preds.tolist()
    y_true = [label2id[r["intent"]] for r in split]
    row_locales = [r["locale"] for r in split]

    out_dir = run_dir / f"eval_{args.partition}{'_smoke' if args.max_samples else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for r, t, p, c in zip(split, y_true, y_pred, confidences.tolist()):
            handle.write(json.dumps(
                {"id": r["id"], "locale": r["locale"], "utt": r["utt"],
                 "true": labels[t], "pred": labels[p], "confidence": round(c, 4)},
                ensure_ascii=False) + "\n")

    scopes = {"overall": list(range(len(split)))}
    for locale in LOCALES:
        scopes[locale] = [i for i, loc in enumerate(row_locales) if loc == locale]

    results: dict = {"partition": args.partition, "run_dir": str(run_dir.name),
                     "n": len(split), "scopes": {}, "top_confused_pairs": {},
                     "within_scenario_error_share": {}}
    for scope, idx in scopes.items():
        yt, yp = [y_true[i] for i in idx], [y_pred[i] for i in idx]
        entry = {"n": len(idx), **compute_metrics(yt, yp)}
        if args.bootstrap:
            entry["ci95"] = bootstrap_ci(yt, yp, n_boot=args.bootstrap)
        results["scopes"][scope] = entry
        if scope == "overall":
            continue
        cm = build_confusion(yt, yp, len(labels))
        write_confusion_csv(out_dir / f"confusion_{scope}.csv", cm, labels)
        results["top_confused_pairs"][scope] = top_confused_pairs(cm, labels, k=15)
        results["within_scenario_error_share"][scope] = within_scenario_error_share(
            yt, yp, labels, intent_to_scenario)
        report = classification_report(yt, yp, labels=list(range(len(labels))),
                                       target_names=labels, output_dict=True, zero_division=0)
        (out_dir / f"per_intent_{scope}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "eval.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"partition={args.partition} n={len(split)} model={run_dir.name}")
    print(f"{'scope':<9}{'n':>6}{'acc':>9}{'macro-F1':>10}   95% CI acc / macro-F1")
    for scope, e in results["scopes"].items():
        ci = e.get("ci95")
        ci_str = (f"[{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] / "
                  f"[{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}]") if ci else "-"
        print(f"{scope:<9}{e['n']:>6}{e['accuracy']:>9.4f}{e['macro_f1']:>10.4f}   {ci_str}")
    for scope in LOCALES:
        share = results["within_scenario_error_share"][scope]
        print(f"{scope}: errors within same scenario: {share:.1%}")
        for pair in results["top_confused_pairs"][scope][:5]:
            print(f"  {pair['true']} -> {pair['pred']}: {pair['count']}")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())