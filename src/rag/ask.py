
#!/usr/bin/env python3
"""Ask the RAG pipeline: retrieve -> rerank -> gate -> LLM -> verified citations."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import DEFAULT_CACHE_DIR, LLMClient, LLMSettings  # noqa: E402
from rag.answer import answer_question  # noqa: E402
from rag.bm25 import BM25Index  # noqa: E402
from rag.chunking import load_chunks  # noqa: E402
from rag.context import CONTEXT_FETCH, DEFAULT_TAU, build_context  # noqa: E402
from rag.dense import DEFAULT_CACHE, BgeM3Encoder, DenseIndex  # noqa: E402
from rag.metrics import DEFAULT_CASES, REPO_ROOT, Case, load_cases  # noqa: E402
from rag.normalize import detect_language  # noqa: E402
from rag.rerank import CrossEncoderReranker  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402



def _status(outcome) -> str:
    if outcome is None:
        return "-"
    if outcome.status == "answered":
        return f"degraded({outcome.degraded_reason})" if outcome.degraded else "answered"
    return f"refused:{outcome.refusal_reason}"


async def run(args: argparse.Namespace) -> None:
    chunks = load_chunks()
    retriever = HybridRetriever(
        BM25Index(chunks), DenseIndex(chunks, BgeM3Encoder(device=args.device), cache_path=DEFAULT_CACHE),
        CrossEncoderReranker(device=args.device), fetch_k=20, rerank_top=20)
    llm = None if args.no_llm else LLMClient(LLMSettings.from_env(), cache_dir=DEFAULT_CACHE_DIR)
    try:
        if args.calibration:
            cases = load_cases(DEFAULT_CASES, {c.doc_id for c in chunks})[: args.limit]
        else:
            cases = [Case("adhoc", detect_language(args.query), "adhoc", args.query, ())]
        rows = []
        for case in cases:
            res = retriever.retrieve(case.query, final_k=CONTEXT_FETCH)
            ctx = build_context(res.final, case.query, tau=args.tau)
            outcome = await answer_question(case.query, ctx, llm) if llm else None
            rows.append((case, ctx, outcome))
            if not args.calibration:
                print(f"gate p={ctx.gate_value} tau={ctx.tau} passed={ctx.passed} "
                      f"context={[c.chunk_id for c in ctx.chunks]} dropped_superseded={ctx.dropped_superseded}")
                if outcome:
                    print(json.dumps(asdict(outcome), ensure_ascii=False, indent=2))
                continue
            label = "ANS" if case.answerable else "UNAN"
            p = f"{ctx.gate_value:.4f}" if ctx.gate_value is not None else "n/a"
            cited = "-"
            if outcome and outcome.status == "answered" and case.answerable:
                cited = "Y" if {c["doc_id"] for c in outcome.citations} & set(case.gold_doc_ids) else "N"
            sup = f" dropped_superseded={ctx.dropped_superseded}" if ctx.dropped_superseded else ""
            print(f"{case.id:<8}{label:<6}{p:>8} {'PASS' if ctx.passed else 'STOP':<5}"
                  f"{_status(outcome):<32}cited_gold={cited}{sup}")
        if args.calibration:
            summarize(rows, args.tau, llm)
    finally:
        if llm:
            await llm.aclose()


def summarize(rows: list, tau: float, llm: LLMClient | None) -> None:
    ans = [r for r in rows if r[0].answerable]
    una = [r for r in rows if not r[0].answerable]
    print(f"\nGate @ tau={tau}: answerable passed {sum(r[1].passed for r in ans)}/{len(ans)}; "
          f"unanswerable stopped {sum(not r[1].passed for r in una)}/{len(una)}")
    if not llm:
        return
    answered = [r for r in ans if r[2].status == "answered"]
    cited = [r for r in answered if {c["doc_id"] for c in r[2].citations} & set(r[0].gold_doc_ids)]
    print(f"Answerable: answered {len(answered)}/{len(ans)}, cited a gold doc {len(cited)}/{len(ans)}, "
          f"degraded {sum(r[2].degraded for r in ans)}")
    print(f"Unanswerable: refused {sum(r[2].status == 'refused' for r in una)}/{len(una)}; "
          f"ANSWERED (bad): {[r[0].id for r in una if r[2].status == 'answered']}")
    print(f"Dropped claims by reason: " + json.dumps(_count(d["reason"] for r in rows for d in r[2].dropped_claims)))
    calls = [r[2].llm for r in rows if r[2].llm and not r[2].llm["cached"]]
    if calls:
        lat = sorted(c["latency_ms"] for c in calls)
        p_tok = [c["prompt_tokens"] or 0 for c in calls]
        c_tok = [c["completion_tokens"] or 0 for c in calls]
        print(f"LLM calls (uncached) {len(calls)}: prompt tokens mean {statistics.mean(p_tok):.0f}, "
              f"completion mean {statistics.mean(c_tok):.0f} (max {max(c_tok)}), total {sum(p_tok) + sum(c_tok)}; "
              f"latency p50 {statistics.median(lat):.0f} ms, max {lat[-1]:.0f} ms; "
              f"retries {sum(c['retries'] for c in calls)}")
    out = REPO_ROOT / "results" / "rag" / f"answers_calibration_{llm.settings.model.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [{"id": c.id, "category": c.category, "gate_value": x.gate_value, "passed": x.passed,
                "dropped_superseded": x.dropped_superseded, "outcome": asdict(o)} for c, x, o in rows]
    out.write_text(json.dumps({"model": llm.settings.model, "tau": tau, "records": records},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def _count(items) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="?")
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="stop after the evidence gate")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    if not args.query and not args.calibration:
        p.error("give a query or --calibration")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()