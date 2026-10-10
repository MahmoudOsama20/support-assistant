#!/usr/bin/env python3
"""Retrieval-only arms on the eval RAG cases (no LLM, no tokens). Ablation = hybrid vs hybrid_rerank.

Run: python src/eval/retrieval_arms.py [--device auto|cpu|cuda] [--out results/eval/retrieval_arms.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("bm25", "dense", "hybrid", "hybrid_rerank")
DEPTH = 20


def doc_ranking(hits: Sequence[Any]) -> list[str]:
    """Chunk hits -> doc ids in rank order, first occurrence only."""
    seen: list[str] = []
    for h in hits:
        d = h.chunk.doc_id
        if d not in seen:
            seen.append(d)
    return seen


def compare_ranks(base: Sequence[int | None], other: Sequence[int | None]) -> dict[str, int]:
    """Per-case paired comparison of first-gold ranks (None = not found = worst)."""
    inf = float("inf")
    out = {"better": 0, "tie": 0, "worse": 0}
    for b, o in zip(base, other, strict=True):
        b, o = inf if b is None else b, inf if o is None else o
        out["better" if o < b else "worse" if o > b else "tie"] += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "eval" / "retrieval_arms.json")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from eval.cases import load_eval_cases
    from rag.bm25 import BM25Index
    from rag.chunking import load_chunks
    from rag.dense import DEFAULT_CACHE, BgeM3Encoder, DenseIndex
    from rag.metrics import first_gold_rank, summarize
    from rag.rerank import CrossEncoderReranker
    from rag.retriever import HybridRetriever

    cases = [c for c in load_eval_cases() if c.gold_doc_ids and c.expected_status == "success"
             and not c.category.startswith("sql_")]
    chunks = load_chunks()
    retriever = HybridRetriever(
        BM25Index(chunks, light_stem=True),
        DenseIndex(chunks, BgeM3Encoder(device=args.device), cache_path=DEFAULT_CACHE),
        CrossEncoderReranker(device=args.device),
        fetch_k=DEPTH, rerank_top=DEPTH,
    )
    retriever.retrieve("warm up")

    ranks: dict[str, dict[str, int | None]] = {a: {} for a in ARMS}
    for c in cases:
        res = retriever.retrieve(c.query, final_k=DEPTH)
        assert res.reranked, "reranker returned nothing"
        arms = {"bm25": res.bm25, "dense": res.dense, "hybrid": res.fused, "hybrid_rerank": res.reranked}
        for arm, hits in arms.items():
            ranks[arm][c.id] = first_gold_rank(doc_ranking(hits), set(c.gold_doc_ids))

    def groups() -> dict[str, list]:
        g: dict[str, list] = {"ALL": cases}
        for c in cases:
            g.setdefault(c.category, []).append(c)
        return g

    report: dict[str, Any] = {"n_cases": len(cases), "groups": {}, "paired_rerank_vs_hybrid": {}}
    for name, cs in groups().items():
        report["groups"][name] = {"n": len(cs), **{arm: summarize([ranks[arm][c.id] for c in cs]) for arm in ARMS}}
    ids = [c.id for c in cases]
    report["paired_rerank_vs_hybrid"] = compare_ranks([ranks["hybrid"][i] for i in ids],
                                                      [ranks["hybrid_rerank"][i] for i in ids])
    report["missed_by_arm"] = {arm: [i for i in ids if ranks[arm][i] is None] for arm in ARMS}
    report["rank_gt1_by_arm"] = {arm: [i for i in ids if ranks[arm][i] not in (None, 1)] for arm in ARMS}

    for name, g in report["groups"].items():
        print(f"\n[{name}] n={g['n']}")
        for arm in ARMS:
            print(f"  {arm:<14}{g[arm]}")
    print(f"\npaired hybrid_rerank vs hybrid: {report['paired_rerank_vs_hybrid']}")
    print(f"missed (gold not in top {DEPTH}): { {a: v for a, v in report['missed_by_arm'].items()} }")
    print(f"gold below rank 1: { {a: v for a, v in report['rank_gt1_by_arm'].items()} }")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()