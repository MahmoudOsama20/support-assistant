"""Score the route model on the hand-written reality-check set (DEV use only).

Usage: python src/route/reality_check.py [--tau 0.95] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.models import ROUTE_DIR, RouteClassifier  # noqa: E402
from agent.policy import DEFAULT_TAU_ROUTE, decide  # noqa: E402
from agent.rules import is_too_short  # noqa: E402

CHECK_PATH = PROJECT_ROOT / "data" / "route" / "reality_check.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "route" / "route_train.jsonl"
OUT_PATH = PROJECT_ROOT / "results" / "route" / "reality_check_v1.json"
EXPECTED_ACTION = {
    "kb_question": "rag",
    "data_lookup": "sql",
    "out_of_scope": "refuse",
    "unsafe_request": "refuse",
    "clarify": "clarify",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_length_diagnostic() -> dict:
    """Evidence for the 'short Arabic question -> out_of_scope' hypothesis."""
    rows = load_jsonl(TRAIN_PATH)
    out: dict = {}
    for lang in ("ar", "en"):
        for label in ("kb_question", "data_lookup", "out_of_scope", "unsafe_request"):
            sel = [r["text"] for r in rows if r["language"] == lang and r["label"] == label]
            words = sorted(len(t.split()) for t in sel)
            short = sum(1 for n in words if n <= 6)
            out[f"{lang}/{label}"] = {
                "n": len(sel),
                "median_words": words[len(words) // 2] if words else None,
                "share_le_6_words": round(short / len(sel), 3) if sel else None,
            }
    ar_oos = [r["text"] for r in rows if r["language"] == "ar" and r["label"] == "out_of_scope"]
    out["ar_oos_starting_kam"] = sum(1 for t in ar_oos if t.startswith("كم"))
    out["ar_oos_total"] = len(ar_oos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU_ROUTE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-dir", type=Path, default=ROUTE_DIR)
    args = ap.parse_args()

    rows = load_jsonl(CHECK_PATH)
    model = RouteClassifier(ROUTE_DIR, args.device)
    results = []
    for r in rows:
        probs = model.predict(r["text"])
        top = max(range(len(probs)), key=probs.__getitem__)
        pred = model.labels[top]
        d = decide(probs, model.labels, args.tau)
        results.append({
            **r,
            "argmax": pred,
            "conf": round(float(probs[top]), 4),
            "action": "clarify" if is_too_short(r["text"]) else d.action,
            "expected_action": EXPECTED_ACTION[r["label"]],
        })

    scored = [x for x in results if x["label"] != "clarify"]
    argmax_ok = sum(x["argmax"] == x["label"] for x in scored)
    action_ok = sum(x["action"] == x["expected_action"] for x in results)
    print(f"argmax acc (non-clarify): {argmax_ok}/{len(scored)}")
    print(f"policy action acc (all, tau={args.tau}): {action_ok}/{len(results)}")

    for lang in ("ar", "en"):
        sub = [x for x in scored if x["language"] == lang]
        print(f"  {lang}: argmax {sum(x['argmax'] == x['label'] for x in sub)}/{len(sub)}")

    conf: Counter = Counter((x["label"], x["argmax"]) for x in scored)
    print("\nconfusion (true -> predicted):")
    for (t, p), n in sorted(conf.items()):
        print(f"  {t:15s} -> {p:15s} {n}")

    print("\nmisses (policy action != expected):")
    for x in results:
        if x["action"] != x["expected_action"]:
            print(f"  {x['id']} [{x['language']}] {x['label']} -> {x['argmax']} "
                  f"({x['conf']}) action={x['action']} :: {x['text']}")

    diag = train_length_diagnostic()
    print("\ntrain diagnostic:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"tau": args.tau, "results": results, "train_diagnostic": diag},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()