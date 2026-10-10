#!/usr/bin/env python
"""Run queries through the full agent (real models, real tools).

  python src/agent/ask_agent.py "query 1" "query 2" [--json] [--tau 0.95] [--device cpu|cuda] [--db PATH]
Dev tip (free-tier quota): $env:LLM_MODEL="openai/gpt-oss-20b"; $env:LLM_MIN_INTERVAL_S="15"
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.build import DEFAULT_DB, build_agent  # noqa: E402
from agent.core import AgentConfig  # noqa: E402


def show(res) -> None:
    print(f"[{res.status}] action={res.action} route={res.route} ({res.route_confidence:.3f}) "
          f"lang={res.language} total={res.timings_ms.get('total', 0):.0f}ms")
    if res.error_code:
        print(f"  error: {res.error_code} (retryable={res.retryable})")
    if res.refusal_reason:
        print(f"  refusal_reason: {res.refusal_reason}")
    if res.answer:
        print(f"  answer: {res.answer}")
    for c in res.citations:
        print(f"  cite: {c.get('doc_id')} {c.get('chunk_id')}")
    if res.sql:
        print(f"  sql: {res.sql} -> rows={res.row_count}")
    active = [k for k, v in res.flags.items() if v]
    print(f"  flags: {active} | intent: {res.intent} ({res.intent_confidence}) | degraded: {res.degraded}")
    print(f"  timings_ms: { {k: round(v, 1) for k, v in res.timings_ms.items()} }")


async def run(queries: list[str], args: argparse.Namespace) -> int:
    bundle = build_agent(config=AgentConfig(tau_route=args.tau), db_path=args.db, device=args.device)
    try:
        for q in queries:
            print(f"\n> {q}")
            res = await bundle.agent.handle(q)
            if args.json:
                print(json.dumps(dataclasses.asdict(res), ensure_ascii=False, indent=2, default=str))
            else:
                show(res)
    finally:
        await bundle.aclose()
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # Arabic on Windows consoles
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("queries", nargs="+")
    p.add_argument("--json", action="store_true")
    p.add_argument("--tau", type=float, default=AgentConfig().tau_route)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    a = p.parse_args()
    return asyncio.run(run(a.queries, a))


if __name__ == "__main__":
    raise SystemExit(main())