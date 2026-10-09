"""Hybrid retrieval: BM25 + dense -> Reciprocal Rank Fusion -> optional cross-encoder rerank."""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.bm25 import Hit


class Searcher(Protocol):
    def search(self, query: str, k: int = 20) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


def rrf_fuse(rankings: Sequence[Sequence[Hit]], *, k: int = 60, top: int = 20) -> list[Hit]:
    """RRF(d) = sum over rankings of 1 / (k + rank). Ranks are 1-based; ties keep first-seen order."""
    scores: dict[str, float] = {}
    chunks: dict = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(cid, hit.chunk)
            order.setdefault(cid, len(order))
    ranked = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))[:top]
    return [Hit(chunks[cid], scores[cid]) for cid in ranked]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    bm25: list[Hit]
    dense: list[Hit]
    fused: list[Hit]
    reranked: list[Hit] | None  # None when no reranker (ablation arm)
    final: list[Hit]            # what the answer step sees
    timings_ms: dict[str, float]


class HybridRetriever:
    def __init__(self, bm25: Searcher, dense: Searcher, reranker: Reranker | None = None, *,
                 fetch_k: int = 20, rerank_top: int = 20) -> None:
        self.bm25, self.dense, self.reranker = bm25, dense, reranker
        self.fetch_k, self.rerank_top = fetch_k, rerank_top

    def retrieve(self, query: str, final_k: int = 4) -> RetrievalResult:
        timings: dict[str, float] = {}

        def run(stage: str, fn):
            start = time.perf_counter()
            out = fn()
            timings[stage] = (time.perf_counter() - start) * 1000
            return out

        bm25_hits = run("bm25", lambda: self.bm25.search(query, self.fetch_k))
        dense_hits = run("dense", lambda: self.dense.search(query, self.fetch_k))
        fused = run("fuse", lambda: rrf_fuse([bm25_hits, dense_hits], top=self.fetch_k))
        reranked = None
        if self.reranker is not None:
            reranked = run("rerank", lambda: self.reranker.rerank(query, fused[:self.rerank_top], final_k))
        final = reranked if reranked is not None else fused[:final_k]
        return RetrievalResult(query, bm25_hits, dense_hits, fused, reranked, final, timings)