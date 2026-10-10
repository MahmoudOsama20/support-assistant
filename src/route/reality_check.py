"""Score a route model on the hand-written reality-check set (DEV use only).

Usage: python src/route/reality_check.py [--model-dir DIR] [--tau 0.95] [--device cpu]
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
OUT_DIR = PROJECT_ROOT / "results" / "route"
EXPECTED_ACTION = {
    "kb_question": "rag",
    "data_lookup": "sql",
    "out_of_scope": "refuse",
    "unsafe_request": "refuse",
    "clarify": "clarify",
}
TAU_GRID = (0.5, 0.8, 0.9, 0.95, 0.98)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def action_for(text: str, probs: list[float], labels: list[str], tau: float) -> str:
    """Pipeline order: the short-query rule runs before the model."""
    return "clarify" if is_too_short(text) else decide(probs, labels, tau).action


def train_length_diagnostic() -> dict:
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROUTE_DIR)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU_ROUTE)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_name = args.model_dir.resolve().parent.name
    out_path = OUT_DIR / f"reality_check_{run_name}.json"
    print(f"model: {args.model_dir} (run {run_name}) | tau {args.tau}")

    rows = load_jsonl(CHECK_PATH)
    model = RouteClassifier(args.model_dir, args.device)
    print(f"temperature: {model.temperature}")
    labels = list(model.labels)

    results, all_probs = [], []
    for r in rows:
        probs = model.predict(r["text"])
        all_probs.append(probs)
        top = max(range(len(probs)), key=probs.__getitem__)
        results.append({
            **r,
            "argmax": labels[top],
            "conf": round(float(probs[top]), 4),
            "action": action_for(r["text"], probs, labels, args.tau),
            "expected_action": EXPECTED_ACTION[r["label"]],
        })

    scored = [x for x in results if x["label"] != "clarify"]
    argmax_ok = sum(x["argmax"] == x["label"] for x in scored)
    action_ok = sum(x["action"] == x["expected_action"] for x in results)
    confident_wrong = [x for x in scored if x["argmax"] != x["label"] and x["conf"] >= 0.95]
    print(f"argmax acc (non-clarify): {argmax_ok}/{len(scored)}")
    print(f"policy action acc (all, tau={args.tau}): {action_ok}/{len(results)}")
    print(f"confident wrong (argmax wrong, conf >= 0.95): {len(confident_wrong)}")
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

    print("\npolicy acc by tau (diagnostic only, NOT used to pick tau):")
    for t in TAU_GRID:
        ok = sum(action_for(r["text"], p, labels, t) == EXPECTED_ACTION[r["label"]]
                 for r, p in zip(rows, all_probs))
        print(f"  tau={t}: {ok}/{len(rows)}")

    diag = train_length_diagnostic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"model_dir": str(args.model_dir), "tau": args.tau,
                    "results": results, "train_diagnostic": diag},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()