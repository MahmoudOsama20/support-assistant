"""p50/p95 latency per route from the JSONL trace.

python src/service/trace_stats.py [logs/traces.jsonl] [--skip-first N] [--exclude-cached] [--json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "logs" / "traces.jsonl"


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]."""
    if not values:
        raise ValueError("percentile of empty list")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(records: list[dict[str, Any]], *, skip_first: int = 0, exclude_cached: bool = False) -> dict[str, Any]:
    recs = records[skip_first:]
    if exclude_cached:
        recs = [r for r in recs if r.get("llm_cached") is not True]
    by_route: dict[str, list[float]] = defaultdict(list)
    all_lat: list[float] = []
    for r in recs:
        lat = r.get("total_latency_ms")
        if lat is None:
            continue
        by_route[str(r.get("route"))].append(float(lat))
        all_lat.append(float(lat))

    def stats(xs: list[float]) -> dict[str, float]:
        return {
            "n": len(xs),
            "p50_ms": round(percentile(xs, 0.50), 1),
            "p95_ms": round(percentile(xs, 0.95), 1),
            "max_ms": round(max(xs), 1),
        }

    groups = {route: stats(xs) for route, xs in sorted(by_route.items())}
    if all_lat:
        groups["ALL"] = stats(all_lat)
    llm_recs = [r for r in recs if r.get("llm_cached") is not None]
    return {
        "groups": groups,
        "status_counts": dict(Counter(r.get("status") for r in recs)),
        "llm_requests": len(llm_recs),
        "llm_cached": sum(1 for r in llm_recs if r.get("llm_cached") is True),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"trace file not found: {path}")
    records, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"warning: skipped {bad} malformed line(s)")
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    p.add_argument("--skip-first", type=int, default=0, help="drop N warm-up requests")
    p.add_argument("--exclude-cached", action="store_true", help="drop requests served from the LLM cache")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out = summarize(load_records(Path(args.path)), skip_first=args.skip_first, exclude_cached=args.exclude_cached)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"{'route':<16}{'n':>5}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}")
    for route, s in out["groups"].items():
        print(f"{route:<16}{s['n']:>5}{s['p50_ms']:>10}{s['p95_ms']:>10}{s['max_ms']:>10}")
    print(f"status: {out['status_counts']}")
    print(f"LLM requests: {out['llm_requests']} (cached: {out['llm_cached']})")
    if out["groups"].get("ALL", {}).get("n", 0) < 20:
        print("note: n < 20, p95 is not meaningful")


if __name__ == "__main__":
    main()