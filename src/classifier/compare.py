#!/usr/bin/env python
"""Step 4: same-row comparison of the fine-tuned models and the LLM baseline.

Reads existing predictions.jsonl files only. Loads no model, runs no new test eval.
Macro-F1 is computed over classes PRESENT in the true labels; invalid LLM
outputs count as wrong and add no phantom class.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
Key = tuple[str, str]  # (id, locale)
SCOPES = ("overall", "ar-SA", "en-US")
_ARABIC = re.compile(r"[\u0600-\u06FF]")


# ---------- loading / alignment ----------
def normalize_pred(row: dict) -> str | None:
    pred = row.get("pred")
    if not isinstance(pred, str) or not pred or row.get("method") == "invalid":
        return None
    return pred


def load_predictions(path: Path) -> dict[Key, dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing predictions file: {path}")
    rows: dict[Key, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (str(r["id"]), r["locale"])
            if key in rows:
                raise ValueError(f"duplicate row {key} in {path}")
            rows[key] = r
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def align(models: dict[str, dict[Key, dict]], keys: list[Key]):
    truth: list[str] = []
    preds: dict[str, list[str | None]] = {n: [] for n in models}
    for k in keys:
        labels = set()
        for name, rows in models.items():
            if k not in rows:
                raise ValueError(f"{name} has no prediction for {k}")
            labels.add(rows[k]["true"])
            preds[name].append(normalize_pred(rows[k]))
        if len(labels) != 1:
            raise ValueError(f"true labels disagree across models for {k}: {labels}")
        truth.append(labels.pop())
    return truth, preds


def encode(truth: list[str], preds: dict[str, list[str | None]]):
    vocab = sorted(set(truth) | {p for ps in preds.values() for p in ps if p is not None})
    l2i = {label: i for i, label in enumerate(vocab)}
    t = np.array([l2i[x] for x in truth])
    p = {n: np.array([l2i[x] if x is not None else -1 for x in ps]) for n, ps in preds.items()}
    return t, p, vocab


# ---------- metrics ----------
def present_metrics(t: np.ndarray, p: np.ndarray, n_classes: int) -> tuple[float, float]:
    """Accuracy and macro-F1 over classes present in t. p == -1 means invalid."""
    acc = float((t == p).mean())
    support = np.bincount(t, minlength=n_classes)
    tp = np.bincount(t[t == p], minlength=n_classes)
    pred_count = np.bincount(p[p >= 0], minlength=n_classes)
    denom = support + pred_count
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return acc, float(f1[support > 0].mean())


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, p)


def _ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_metrics(t, preds, ids, n_classes, n_boot, seed):
    """Cluster bootstrap: resample ids (ar/en rows of one id stay together)."""
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(ids):
        groups.setdefault(k, []).append(i)
    group_list = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    res = {n: {"acc": np.empty(n_boot), "f1": np.empty(n_boot)} for n in preds}
    for b in range(n_boot):
        pick = rng.integers(0, len(group_list), len(group_list))
        idx = np.concatenate([group_list[j] for j in pick])
        for n, p in preds.items():
            res[n]["acc"][b], res[n]["f1"][b] = present_metrics(t[idx], p[idx], n_classes)
    return res


def scope_report(t, preds, ids, n_classes, n_boot, seed, pairs):
    boot = bootstrap_metrics(t, preds, ids, n_classes, n_boot, seed)
    out = {"n": int(len(t)), "models": {}, "diffs": {}}
    point = {n: present_metrics(t, p, n_classes) for n, p in preds.items()}
    for n, (acc, f1) in point.items():
        out["models"][n] = {
            "accuracy": acc, "macro_f1": f1,
            "ci95_accuracy": _ci(boot[n]["acc"]), "ci95_macro_f1": _ci(boot[n]["f1"]),
        }
    for a, b in pairs:
        b_wins, a_wins, p = mcnemar_exact(preds[a] == t, preds[b] == t)
        out["diffs"][f"{a}-{b}"] = {
            "accuracy_diff": point[a][0] - point[b][0],
            "ci95_accuracy_diff": _ci(boot[a]["acc"] - boot[b]["acc"]),
            "macro_f1_diff": point[a][1] - point[b][1],
            "ci95_macro_f1_diff": _ci(boot[a]["f1"] - boot[b]["f1"]),
            "mcnemar": {f"{a}_right_{b}_wrong": b_wins, f"{b}_right_{a}_wrong": a_wins, "p_exact": p},
        }
    return out


def build_reports(keys, truth, preds_labels, n_boot, seed, pairs):
    t, p, vocab = encode(truth, preds_labels)
    ids = [k[0] for k in keys]
    locs = np.array([k[1] for k in keys])
    reports = {}
    for scope in SCOPES:
        sel = np.arange(len(keys)) if scope == "overall" else np.flatnonzero(locs == scope)
        reports[scope] = scope_report(
            t[sel], {n: v[sel] for n, v in p.items()}, [ids[i] for i in sel],
            len(vocab), n_boot, seed, pairs,
        )
    return reports, t, p, vocab


# ---------- extra analyses ----------
def quirky_qa_share(t: np.ndarray, p: np.ndarray, vocab: list[str]) -> dict:
    special = np.array([v == "general_quirky" or v.startswith("qa_") for v in vocab])
    wrong = p != t
    touches = special[t] | np.where(p >= 0, special[np.maximum(p, 0)], False)
    n_wrong = int(wrong.sum())
    return {
        "n_errors": n_wrong,
        "share_of_errors_touching_quirky_or_qa": float((wrong & touches).sum() / n_wrong) if n_wrong else None,
        "share_of_rows_with_true_label_quirky_or_qa": float(special[t].mean()),
    }


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


def arabic_buckets(keys, utts: dict[Key, str], t, preds) -> dict:
    sel = np.array([i for i, k in enumerate(keys) if k[1] == "ar-SA"])
    flags = np.array([has_arabic(utts[keys[i]]) for i in sel])
    out = {}
    for name, m in (("has_arabic", flags), ("no_arabic", ~flags)):
        idx = sel[m]
        out[name] = {
            "n": int(len(idx)),
            "accuracy": {n: (float((p[idx] == t[idx]).mean()) if len(idx) else None) for n, p in preds.items()},
        }
    return out


def llm_list_price(llm_dir: Path, cfg_path: Path) -> dict:
    summary = json.loads((llm_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pin, pout = cfg.get("price_per_1m_input_usd"), cfg.get("price_per_1m_output_usd")
    mp, mc = summary["tokens"]["mean_prompt"], summary["tokens"]["mean_completion"]
    out = {"mean_prompt_tokens": mp, "mean_completion_tokens": mc, "price_date": cfg.get("price_date")}
    if pin is None or pout is None:
        out["list_price_usd_per_request"] = None
        out["note"] = "prices not set in configs/llm_baseline.yaml; not computed"
    else:
        out["list_price_usd_per_request"] = (mp * pin + mc * pout) / 1e6
        out["note"] = "list-price equivalent; free-tier actual cost is $0"
    return out


# ---------- printing ----------
def print_reports(title: str, reports: dict) -> None:
    print(f"\n===== {title} =====")
    for scope, r in reports.items():
        print(f"[{scope}] n={r['n']}")
        for n, m in r["models"].items():
            a, f = m["ci95_accuracy"], m["ci95_macro_f1"]
            print(f"  {n:<5} acc {m['accuracy']:.4f} [{a[0]:.4f}, {a[1]:.4f}]   "
                  f"macro-F1 {m['macro_f1']:.4f} [{f[0]:.4f}, {f[1]:.4f}]")
        for name, d in r["diffs"].items():
            a, f, mc = d["ci95_accuracy_diff"], d["ci95_macro_f1_diff"], d["mcnemar"]
            print(f"  {name}: acc {d['accuracy_diff']:+.4f} [{a[0]:+.4f}, {a[1]:+.4f}]   "
                  f"F1 {d['macro_f1_diff']:+.4f} [{f[0]:+.4f}, {f[1]:+.4f}]   "
                  f"McNemar p={mc['p_exact']:.4g} {({k: v for k, v in mc.items() if k != 'p_exact'})}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlmr-run", type=Path, default=ROOT / "results/runs/xlmr-base-e10")
    ap.add_argument("--e5-run", type=Path, default=ROOT / "results/runs/e5-small-e15")
    ap.add_argument("--llm-dir", type=Path, default=ROOT / "results/llm_baseline/allam-2-7b/test")
    ap.add_argument("--llm-config", type=Path, default=ROOT / "configs/llm_baseline.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/comparison")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    xl = load_predictions(args.xlmr_run / "eval_test" / "predictions.jsonl")
    e5 = load_predictions(args.e5_run / "eval_test" / "predictions.jsonl")
    llm = load_predictions(args.llm_dir / "predictions.jsonl")
    if set(xl) != set(e5):
        raise ValueError("XLM-R and E5 test predictions cover different rows")
    utts = {k: r["utt"] for k, r in xl.items()}
    result: dict = {}

    # A) full test, fine-tuned pair
    keys = sorted(xl)
    truth, preds = align({"xlmr": xl, "e5": e5}, keys)
    rep, t, p, vocab = build_reports(keys, truth, preds, args.bootstrap, args.seed, [("e5", "xlmr")])
    result["full_test_paired"] = rep
    result["full_test_error_share"] = {n: quirky_qa_share(t, v, vocab) for n, v in p.items()}
    result["full_test_ar_script_buckets"] = arabic_buckets(keys, utts, t, p)
    print_reports("A) FULL TEST, fine-tuned pair (E5 minus XLM-R)", rep)

    # B) the LLM's sampled rows, all three models
    skeys = sorted(llm)
    missing = [k for k in skeys if k not in xl]
    if missing:
        raise ValueError(f"{len(missing)} LLM rows not found in fine-tuned predictions, e.g. {missing[:3]}")
    truth, preds = align({"xlmr": xl, "e5": e5, "llm": llm}, skeys)
    pairs = [("e5", "llm"), ("xlmr", "llm"), ("e5", "xlmr")]
    rep, t, p, vocab = build_reports(skeys, truth, preds, args.bootstrap, args.seed, pairs)
    result["llm_sample_same_rows"] = rep
    result["llm_sample_error_share"] = {n: quirky_qa_share(t, v, vocab) for n, v in p.items()}
    result["llm_sample_ar_script_buckets"] = arabic_buckets(skeys, utts, t, p)
    result["llm_cost"] = llm_list_price(args.llm_dir, args.llm_config)
    print_reports("B) SAME 1,000 ROWS, three models", rep)

    print("\n===== error share touching general_quirky / qa_* =====")
    for label in ("full_test_error_share", "llm_sample_error_share"):
        print(label, json.dumps(result[label], indent=2))
    print("\n===== ar-SA accuracy by Arabic-script presence =====")
    for label in ("full_test_ar_script_buckets", "llm_sample_ar_script_buckets"):
        print(label, json.dumps(result[label], indent=2))
    print("\nLLM cost:", json.dumps(result["llm_cost"], indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()