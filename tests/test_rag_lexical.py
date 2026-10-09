from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

from rag.bm25 import BM25Index
from rag.chunking import DEFAULT_DOCS_DIR, Chunk, chunk_doc, load_chunks
from rag.normalize import normalize_text, tokenize

CANARY = "LOYALTY-OVERRIDE-4471"
FRONT = ('---\ndoc_id: kb_900\ntitle: Demo Title\nlanguage: en\ncategory: x\ntopic: demo\n'
         'version: "1.0"\neffective_date: "2026-01-01"\nstatus: active\nsource: synthetic\n---\n')


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return load_chunks()


@pytest.fixture(scope="module")
def kb_index(chunks: list[Chunk]) -> BM25Index:
    return BM25Index(chunks)


# ---- normalization ----
def test_arabic_normalization() -> None:
    assert normalize_text("أَحْمَد إلى آخر") == "احمد الي اخر"
    assert normalize_text("الخـــط") == "الخط"


def test_digits_and_thousands() -> None:
    assert normalize_text("١٢٣") == "123"
    assert normalize_text("5,000 EGP") == "5000 egp"
    assert normalize_text("٥٬٠٠٠") == "5000"
    assert normalize_text("1,000,000") == "1000000"
    assert "1.5%" in normalize_text("fee 1.5%")


def test_ta_marbuta_flag() -> None:
    assert normalize_text("مدة") == "مده"
    assert normalize_text("مدة", fold_ta_marbuta=False) == "مدة"


def test_normalize_is_idempotent() -> None:
    text = "إِلى ٥٬٠٠٠ جنيهٍ — Nile Wallet 1.5%  "
    assert normalize_text(normalize_text(text)) == normalize_text(text)


def test_tokenize_basic() -> None:
    assert tokenize("Fee: 1.5% (min 50 EGP)") == ["fee", "1.5", "min", "50", "egp"]


def test_light_stem_is_arabic_only_and_optional() -> None:
    assert tokenize("للتحويل الدولي") == ["تحويل", "دولي"]
    assert tokenize("للتحويل الدولي", light_stem=False) == ["للتحويل", "الدولي"]
    assert tokenize("Alpha") == ["alpha"]


# ---- chunking ----
def test_chunk_count_matches_sections_plus_preambles(chunks: list[Chunk]) -> None:
    expected = 0
    for p in DEFAULT_DOCS_DIR.glob("kb_*.md"):
        body = p.read_text(encoding="utf-8").replace("\r\n", "\n").split("---\n", 2)[2]
        lines = body.split("\n")
        expected += sum(1 for line in lines if line.startswith("## "))
        first = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
        if any(line.strip() and not line.startswith("# ") for line in lines[:first]):
            expected += 1  # intro text before the first '## ' becomes its own chunk
    assert len(chunks) == expected


def test_only_fee_schedules_have_a_preamble_chunk(chunks: list[Chunk]) -> None:
    assert {c.doc_id for c in chunks if c.heading == c.title} == {"kb_005", "kb_006"}


def test_chunk_ids_unique_and_sequential(chunks: list[Chunk]) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"kb_\d{3}#c[1-9]\d*", i) for i in ids)
    per_doc: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        per_doc[c.doc_id].append(int(c.chunk_id.split("#c")[1]))
    assert all(nums == list(range(1, len(nums) + 1)) for nums in per_doc.values())
    assert all(c.index_text.startswith(c.title) for c in chunks)


def test_status_and_injection_flags(chunks: list[Chunk]) -> None:
    assert [c.doc_id for c in chunks if CANARY in c.text] == ["kb_032"]
    assert all(c.injection_test for c in chunks if c.doc_id == "kb_032")
    assert not any(c.injection_test for c in chunks if c.doc_id != "kb_032")
    assert {c.status for c in chunks if c.doc_id == "kb_031"} == {"superseded"}
    assert {c.status for c in chunks if c.doc_id != "kb_031"} == {"active"}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_preamble_and_crlf(tmp_path: Path, newline: str) -> None:
    body = "# Demo Title\nIntro line.\n\n## First\nBody one.\n\n## Second\nBody two.\n"
    path = tmp_path / "kb_900.md"
    path.write_bytes((FRONT + body).replace("\n", newline).encode("utf-8"))
    got = [(c.chunk_id, c.heading, c.text) for c in chunk_doc(path)]
    assert got == [("kb_900#c1", "Demo Title", "Intro line."),
                   ("kb_900#c2", "First", "Body one."),
                   ("kb_900#c3", "Second", "Body two.")]


def test_missing_front_matter_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "kb_900.md"
    path.write_text(FRONT.replace("doc_id: kb_900\n", "") + "## A\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        chunk_doc(path)


# ---- BM25 ----
def _mk(i: int, text: str) -> Chunk:
    return Chunk(f"d{i}#c1", f"d{i}", f"title{i}", f"head{i}", text, "en", "t", "active", "2026-01-01", "1.0")


def test_bm25_ranking_and_filters() -> None:
    corpus = [_mk(0, "apple banana"), _mk(1, "banana cherry")] + [_mk(i, f"filler{i}") for i in range(2, 10)]
    index = BM25Index(corpus)
    assert index.search("apple banana")[0].chunk.chunk_id == "d0#c1"
    assert [h.chunk.chunk_id for h in index.search("apple")] == ["d0#c1"]  # zero-score chunks dropped
    assert index.search("zzz") == []
    assert index.search("") == []
    assert len(index.search("banana", k=1)) == 1


def test_arabic_query_finds_arabic_doc(kb_index: BM25Index) -> None:
    docs = [h.chunk.doc_id for h in kb_index.search("ما هي رسوم التحويل الدولي", k=3)]
    assert "kb_018" in docs


def test_english_query_finds_english_doc(kb_index: BM25Index) -> None:
    docs = [h.chunk.doc_id for h in kb_index.search("how long does a refund to my bank card take", k=3)]
    assert "kb_011" in docs