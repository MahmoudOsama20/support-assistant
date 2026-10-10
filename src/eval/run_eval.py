#!/usr/bin/env python3
"""Run the 70-case eval in process (agent.handle + to_response), score it, write results/eval/<name>.json.

Pass 1 = real run (LLM calls paced by LLM_MIN_INTERVAL_S, cache populated).
Pass 2 = rerun from the LLM cache -> pipeline latency without LLM wait or pacing.
estimated_total = pass-2 wall time + pass-1 LLM time (an estimate).

Run: python src/eval/run_eval.py --name final-120b [--ids a,b] [--categories x,y] [--limit N]
     [--no-warmup] [--no-second-pass] [--timeout 60] [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.schemas import RESPONSE_ADAPTER, new_request_id, to_response  # noqa: E402
from eval.cases import DEFAULT_CASES, EvalCase, load_eval_cases  # noqa: E402
from eval.scoring import aggregate, latency_summary, llm_time_ms, run_gold, score_case  # noqa: E402
from service.app import error_response  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "db" / "support.db"
RESULTS_DIR = ROOT / "results" / "eval"
WARMUP_QUERIES = ("what's the weather like in Cairo", "What is the fee for a virtual card?")


async def run_one(agent: Any, query: str, timeout_s: float) -> tuple[dict, dict, float]:
    rid = new_request_id()
    t0 = time.perf_counter()
    result = None
    try:
        result = await asyncio.wait_for(agent.handle(query), timeout=timeout_s)
        response = to_response(result, rid)
    except TimeoutError:
        response = error_response(rid, query, "timeout", "The request took too long.", True)
    except Exception as exc:
        print(f"  crash: {type(exc).__name__}: {exc}")
        response = error_response(rid, query, "internal_error", "Internal error.", False)
    wall_ms = (time.perf_counter() - t0) * 1000
    payload = RESPONSE_ADAPTER.dump_python(response, mode="json")
    return payload, dict(getattr(result, "detail", None) or {}), wall_ms


async def run_pass(agent: Any, cases: list[EvalCase], *, timeout_s: float = 60.0,
                   max_consecutive_errors: int = 3, on_result=None) -> tuple[list[dict], bool]:
    records: list[dict] = []
    streak = 0
    for case in cases:
        payload, detail, wall_ms = await run_one(agent, case.query, timeout_s)
        records.append({"case": case, "payload": payload, "detail": detail, "wall_ms": wall_ms})
        if on_result:
            on_result(case, payload, wall_ms)
        streak = streak + 1 if payload.get("status") == "error" else 0
        if streak >= max_consecutive_errors:
            return records, False  # quota/outage: stop spending time
    return records, True


def select_cases(cases: list[EvalCase], ids: str | None, categories: str | None, limit: int | None) -> list[EvalCase]:
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    if categories:
        cats = {c.strip() for c in categories.split(",")}
        cases = [c for c in cases if c.category in cats]
    return cases[:limit] if limit else cases


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def print_summary(agg: dict[str, Any], lat: dict[str, Any]) -> None:
    def show(name: str) -> None:
        v = agg[name]
        print(f"  {name:<30}{v['k']:>3}/{v['n']:<3} {v['rate']}")

    print("\n== summary ==")
    for name in ("task_success", "sql_lenient", "sql_strict", "over_refusal", "over_clarify", "error_rate",
                 "refusal_correct", "clarify_correct", "unanswerable_answered", "safety_not_refused",
                 "forbidden_string_leaks", "indirect_injection_obeyed", "citation_gold_given_answered",
                 "gold_doc_in_context", "routing_accuracy_diagnostic"):
        show(name)
    print("  by category:")
    for cat, v in agg["by_category"].items():
        print(f"    {cat:<20}{v['k']:>3}/{v['n']:<3} {v['rate']}")
    if lat:
        print("\n== latency (ms; estimated_total = cached pipeline + recorded LLM time) ==")
        print(f"  {'route':<16}{'n':>4}{'est p50':>10}{'est p95':>10}{'pipe p50':>10}{'pipe p95':>10}{'llm p50':>10}")
        for route, g in lat.items():
            e, p, m = g["estimated_total"], g["pipeline"], g["llm"]
            print(f"  {route:<16}{e['n']:>4}{e['p50_ms']:>10}{e['p95_ms']:>10}{p['p50_ms']:>10}{p['p95_ms']:>10}"
                  f"{(m['p50_ms'] if m else '-'):>10}")


async def amain(args: argparse.Namespace) -> None:
    out_path = RESULTS_DIR / f"{args.name}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists (scored run is spent); use --force only for a deliberate rerun.")
    cases = select_cases(load_eval_cases(args.cases), args.ids, args.categories, args.limit)
    if not cases:
        raise SystemExit("no cases selected")
    gold = {c.id: run_gold(args.db, c.gold_sql) for c in cases if c.gold_sql}

    from agent.build import build_agent, pick_device  # heavy: loads models

    bundle = build_agent(db_path=args.db, device=pick_device(), with_intent=True)
    model = os.getenv("LLM_MODEL") or "(default)"
    interval = float(os.getenv("LLM_MIN_INTERVAL_S", "0") or 0)
    print(f"{len(cases)} cases | LLM_MODEL={model} | LLM_MIN_INTERVAL_S={interval}")

    def on_result(case: EvalCase, payload: dict, wall_ms: float) -> None:
        print(f"  {case.id:<22}{payload.get('status'):<9}{str(payload.get('route')):<15}{wall_ms:>8.0f} ms")

    try:
        if not args.no_warmup:
            for q in WARMUP_QUERIES:
                try:
                    await bundle.agent.handle(q)
                except Exception as exc:
                    print(f"warm-up failed ({type(exc).__name__}: {exc}); continuing")
        print("-- pass 1 --")
        p1, complete = await run_pass(bundle.agent, cases, timeout_s=args.timeout, on_result=on_result)
        p2 = None
        if complete and not args.no_second_pass:
            print("-- pass 2 (from cache) --")
            p2, _ = await run_pass(bundle.agent, cases, timeout_s=args.timeout)
    finally:
        await bundle.aclose()

    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(p1):
        case = rec["case"]
        s = score_case(case, rec["payload"], rec["detail"], gold.get(case.id))
        s.update(query=case.query, wall_ms_pass1=round(rec["wall_ms"], 1),
                 llm_ms=round(llm_time_ms(rec["detail"]), 1),
                 llm_cached=(rec["detail"].get("llm") or {}).get("cached"),
                 payload=rec["payload"], detail=rec["detail"])
        if p2 is not None:
            s["wall_ms_pass2"] = round(p2[i]["wall_ms"], 1)
            s["status_pass2"] = p2[i]["payload"].get("status")
        rows.append(s)

    agg, lat = aggregate(rows), latency_summary(rows)
    changed = [r["id"] for r in rows if r.get("status_pass2") not in (None, r["status"])]
    paced = [r["id"] for r in rows if interval and r["llm_ms"] >= 0.9 * interval * 1000]
    meta = {
        "name": args.name, "complete": complete, "n_cases": len(cases), "subset": len(cases) != 70,
        "llm_model": model, "llm_min_interval_s": interval, "commit": git_commit(),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status_changed_pass2": changed, "llm_ms_may_include_pacing": paced,
    }
    target = out_path if complete else out_path.with_suffix(".partial.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"meta": meta, "aggregate": agg, "latency": lat, "cases": rows},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nfailed cases: {[r['id'] for r in rows if not r['passed']]}")
    print_summary(agg, lat)
    if changed:
        print(f"WARNING: status differs between pass 1 and pass 2 for {changed}")
    if paced:
        print(f"WARNING: llm_ms >= 90% of the pacing interval for {paced}: pacing may be inside the LLM time")
    if not complete:
        print("INCOMPLETE: aborted after consecutive errors (quota/outage?). Partial file written; rerun later (cache reuses done calls).")
    print(f"\nwrote {target}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ids")
    p.add_argument("--categories")
    p.add_argument("--limit", type=int)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-second-pass", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()