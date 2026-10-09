import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "route" / "reality_check.jsonl"
LABELS = {"kb_question", "data_lookup", "out_of_scope", "unsafe_request", "clarify"}


def _rows() -> list[dict]:
    with PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_schema_and_uniqueness() -> None:
    rows = _rows()
    assert len(rows) == 42
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["text"] for r in rows}) == len(rows)
    for r in rows:
        assert r["label"] in LABELS
        assert r["language"] in {"ar", "en"}
        assert r["text"].strip()


def test_no_exact_overlap_with_route_train() -> None:
    train = PATH.parent / "route_train.jsonl"
    with train.open(encoding="utf-8") as f:
        seen = {json.loads(line)["text"].strip().lower() for line in f if line.strip()}
    assert not {r["text"].strip().lower() for r in _rows()} & seen