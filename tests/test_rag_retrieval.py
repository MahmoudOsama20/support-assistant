from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag.bm25 import Hit
from rag.chunking import Chunk, load_chunks
from rag.dense import DenseIndex, VectorIndex
from rag.metrics import DEFAULT_CASES, auc, first_gold_rank, load_cases, summarize
from rag.retriever import HybridRetriever, rrf_fuse


def _mk(i: int, text: str = "x") -> Chunk:
    return Chunk(f"d{i}#c1", f"d{i}", f"title{i}", f"head{i}", text, "en", "t", "active", "2026-01-01", "1.0")


def _hit(i: int, score: float = 0.0) -> Hit:
    return Hit(_mk(i), score)


class FakeEncoder:
    name = "fake"
    max_length = 8

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, sum(map(ord, text)) % 4] = 1.0
        return out


class Fixed:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 20) -> list[Hit]:
        return self.hits[:k]


class ReverseReranker:
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return [Hit(h.chunk, float(i)) for i, h in enumerate(reversed(hits))][:top_n]


def test_first_gold_rank() -> None:
    assert first_gold_rank(["a", "b", "c"], {"c", "b"}) == 2
    assert first_gold_rank(["a"], {"z"}) is None
    assert first_gold_rank([], {"z"}) is None


def test_summarize() -> None:
    s = summarize([1, 3, None, 6])
    assert s == pytest.approx({"recall@1": 0.25, "recall@3": 0.5, "recall@5": 0.5,
                               "mrr": (1 + 1 / 3 + 1 / 6) / 4})
    with pytest.raises(ValueError):
        summarize([])


def test_auc() -> None:
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    with pytest.raises(ValueError):
        auc([], [1.0])


def test_vector_index() -> None:
    index = VectorIndex(np.eye(3, dtype=np.float32))
    assert index.search(np.array([0, 1, 0], dtype=np.float32), 2)[0] == (1, 1.0)
    tied = index.search(np.array([1, 1, 0], dtype=np.float32), 3)
    assert [i for i, _ in tied] == [0, 1, 2]
    assert len(index.search(np.array([1, 0, 0], dtype=np.float32), 10)) == 3
    with pytest.raises(ValueError):
        index.search(np.zeros(5, dtype=np.float32), 1)


def test_rrf_fuse() -> None:
    a, b, c, d = (_hit(i) for i in range(4))
    fused = rrf_fuse([[a, b, c], [b, a, d]], k=60, top=10)
    assert [h.chunk.chunk_id for h in fused] == ["d0#c1", "d1#c1", "d2#c1", "d3#c1"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2].score == pytest.approx(1 / 63)
    assert len(rrf_fuse([[a, b, c]], top=2)) == 2
    assert rrf_fuse([[], []]) == []


def test_dense_cache_reused_and_invalidated(tmp_path: Path) -> None:
    cache = tmp_path / "idx.npz"
    chunks = [_mk(i, f"text {i}") for i in range(3)]
    first = FakeEncoder()
    index = DenseIndex(chunks, first, cache_path=cache)
    assert first.calls == 1
    scores = [h.score for h in index.search("text 1", k=3)]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)
    second = FakeEncoder()
    DenseIndex(chunks, second, cache_path=cache)
    assert second.calls == 0  # cache hit
    third = FakeEncoder()
    DenseIndex(chunks[:2] + [_mk(2, "different text")], third, cache_path=cache)
    assert third.calls == 1  # chunk text changed -> stale cache rebuilt


def test_retriever_with_and_without_reranker() -> None:
    hits = [_hit(i) for i in range(5)]
    plain = HybridRetriever(Fixed(hits), Fixed(hits)).retrieve("q", final_k=3)
    assert plain.reranked is None and plain.final == plain.fused[:3]
    assert set(plain.timings_ms) == {"bm25", "dense", "fuse"}
    reranked = HybridRetriever(Fixed(hits), Fixed(hits), ReverseReranker(), rerank_top=4).retrieve("q", final_k=2)
    assert [h.chunk.chunk_id for h in reranked.final] == ["d3#c1", "d2#c1"]
    assert "rerank" in reranked.timings_ms


def test_calibration_file_is_valid() -> None:
    cases = load_cases(DEFAULT_CASES, {c.doc_id for c in load_chunks()})
    assert len(cases) == 24
    assert sum(c.answerable for c in cases) == 16
    for case in cases:
        letters = [ch for ch in case.query if ch.isalpha()]
        arabic = sum("\u0600" <= ch <= "\u06ff" for ch in letters)
        assert (arabic / len(letters) > 0.5) == (case.language == "ar"), case.id


def test_load_cases_rejects_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    row = {"id": "a", "language": "en", "category": "x", "query": "q", "gold_doc_ids": ["kb_001"]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path, {"kb_001"})
    path.write_text(json.dumps({**row, "gold_doc_ids": ["kb_999"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_cases(path, {"kb_001"})