from types import SimpleNamespace

from eval.retrieval_arms import compare_ranks, doc_ranking


def hit(doc_id):
    return SimpleNamespace(chunk=SimpleNamespace(doc_id=doc_id))


def test_doc_ranking_dedupes_in_order():
    hits = [hit("kb_003"), hit("kb_004"), hit("kb_003"), hit("kb_001")]
    assert doc_ranking(hits) == ["kb_003", "kb_004", "kb_001"]
    assert doc_ranking([]) == []


def test_compare_ranks_counts_better_tie_worse():
    base = [1, 3, None, 2, None]
    other = [1, 1, 2, 4, None]
    assert compare_ranks(base, other) == {"better": 2, "tie": 2, "worse": 1}


def test_compare_ranks_requires_equal_lengths():
    import pytest
    with pytest.raises(ValueError):
        compare_ranks([1], [1, 2])