#!/usr/bin/env python3
"""Latency from the warm LLM cache: pipeline wall time + the recorded real LLM time (no tokens, no pacing).

Requires the SAME LLM_MODEL as the scored run so every call is a cache hit.
Run: $env:LLM_MODEL="openai/gpt-oss-120b"; python src/eval/latency_run.py --name latency-120b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cases import DEFAULT_CASES, load_eval_cases  # noqa: E402
from eval.run_eval import DEFAULT_DB, RESULTS_DIR, WARMUP_QUERIES, run_one  # noqa: E402
from eval.scoring import latency_summary  # noqa: E402
from sqltool.llm_adapter import LLM_TRACE  # noqa: E402


def llm_ms_of(trace: list[float], detail: dict[str, Any] | None) -> float:
    """SQL: sum of recorded reply latencies. RAG: detail.llm.latency_ms. Otherwise 0."""
    if trace:
        return float(sum(trace))
    llm = (detail or {}).get("llm")
    if isinstance(llm, dict) and llm.get("latency_ms") is not None:
        return float(llm["latency_ms"])
    return 0.0


async def measure(agent: Any, cases: list, timeout_s: float) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        trace: list[float] = []
        token = LLM_TRACE.set(trace)
        try:
            payload, detail, wall_ms = await run_one(agent, case.query, timeout_s)
        finally:
            LLM_TRACE.reset(token)
        rows.append({"id": case.id, "route": payload.get("route"), "status": payload.get("status"),
                     "wall_ms_pass2": round(wall_ms, 1),  # key name reused by latency_summary
                     "llm_ms": round(llm_ms_of(trace, detail), 1)})
    return rows


async def amain(args: argparse.Namespace) -> None:
    out = RESULTS_DIR / f"{args.name}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; use --force to overwrite.")
    cases = load_eval_cases(args.cases)
    from agent.build import build_agent, pick_device

    bundle = build_agent(db_path=args.db, device=pick_device(), with_intent=True)
    try:
        for q in WARMUP_QUERIES:
            await bundle.agent.handle(q)
        await measure(bundle.agent, cases, args.timeout)  # pass 1: warm-up, discarded
        rows = await measure(bundle.agent, cases, args.timeout)
    finally:
        await bundle.aclose()

    mismatched: list[str] = []
    if args.reference.is_file():
        ref = {r["id"]: r["status"] for r in json.loads(args.reference.read_text(encoding="utf-8"))["cases"]}
        mismatched = [r["id"] for r in rows if ref.get(r["id"]) != r["status"]]
    lat = latency_summary(rows)
    meta = {"name": args.name, "llm_model": os.getenv("LLM_MODEL") or "(default)", "n": len(rows),
            "status_mismatch_vs_reference": mismatched}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "latency": lat, "cases": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"{'route':<16}{'n':>4}{'est p50':>10}{'est p95':>10}{'pipe p50':>10}{'pipe p95':>10}{'llm p50':>10}")
    for route, g in lat.items():
        e, p, m = g["estimated_total"], g["pipeline"], g["llm"]
        print(f"{route:<16}{e['n']:>4}{e['p50_ms']:>10}{e['p95_ms']:>10}{p['p50_ms']:>10}{p['p95_ms']:>10}"
              f"{(m['p50_ms'] if m else '-'):>10}")
    if mismatched:
        print(f"WARNING: status differs from the reference run for {mismatched} (cache miss or nondeterminism)")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--reference", type=Path, default=RESULTS_DIR / "final-120b.json")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()