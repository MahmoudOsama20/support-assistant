"""Eval case file: loader, validator and answer-matching helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "eval" / "cases.jsonl"

EXPECTED_COUNTS = {
    "rag_en": 11, "rag_ar": 11, "rag_cross": 6,
    "sql_en": 8, "sql_ar": 8,
    "kb_unanswerable": 8, "db_unanswerable": 3, "ambiguous": 4,
    "injection_direct": 5, "sql_destructive": 3, "injection_indirect": 2, "prompt_leak": 1,
}
STATUSES = {"success", "refused", "clarify"}
ROUTES = {"kb_question", "data_lookup", "out_of_scope", "unsafe_request", "none"}
REASONS = {"no_evidence", "out_of_scope", "unsafe_request", "policy"}
LANGUAGES = {"ar", "en"}
SQL_CATEGORIES = {"sql_en", "sql_ar"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    language: str
    category: str
    query: str
    expected_route: str
    expected_status: str
    expected_reason: str | None
    gold_doc_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    must_include: tuple[str, ...]
    must_not_include: tuple[str, ...]
    gold_sql: str | None


def normalize_query(text: str) -> str:
    return " ".join(text.split()).lower()


def _parse(row: dict, line_no: int) -> EvalCase:
    def fail(msg: str) -> None:
        raise ValueError(f"cases line {line_no} ({row.get('id', '?')}): {msg}")

    required = ["id", "language", "category", "query", "expected_route", "expected_status",
                "expected_reason", "gold_doc_ids", "gold_chunk_ids", "must_include",
                "must_not_include", "gold_sql"]
    missing = [k for k in required if k not in row]
    if missing:
        fail(f"missing fields {missing}")
    if row["language"] not in LANGUAGES:
        fail(f"bad language {row['language']!r}")
    if row["category"] not in EXPECTED_COUNTS:
        fail(f"unknown category {row['category']!r}")
    if not str(row["query"]).strip():
        fail("empty query")
    if row["expected_route"] not in ROUTES:
        fail(f"bad expected_route {row['expected_route']!r}")
    if row["expected_status"] not in STATUSES:
        fail(f"bad expected_status {row['expected_status']!r}")
    if row["expected_reason"] is not None and row["expected_reason"] not in REASONS:
        fail(f"bad expected_reason {row['expected_reason']!r}")
    if (row["expected_status"] == "refused") != (row["expected_reason"] is not None):
        fail("expected_reason must be set exactly when expected_status is refused")
    return EvalCase(
        id=row["id"], language=row["language"], category=row["category"], query=row["query"],
        expected_route=row["expected_route"], expected_status=row["expected_status"],
        expected_reason=row["expected_reason"],
        gold_doc_ids=tuple(row["gold_doc_ids"]), gold_chunk_ids=tuple(row["gold_chunk_ids"]),
        must_include=tuple(row["must_include"]), must_not_include=tuple(row["must_not_include"]),
        gold_sql=row["gold_sql"],
    )


def load_eval_cases(path: Path = DEFAULT_CASES) -> list[EvalCase]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run python src/eval/build_cases.py")
    cases: list[EvalCase] = []
    ids: set[str] = set()
    queries: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = _parse(json.loads(line), n)
        if case.id in ids:
            raise ValueError(f"duplicate id {case.id}")
        key = normalize_query(case.query)
        if key in queries:
            raise ValueError(f"duplicate query in {case.id}")
        ids.add(case.id)
        queries.add(key)
        cases.append(case)
    return cases


# --- answer matching (used by the 14b runner) ---------------------------------------------

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize_for_match(text: str) -> str:
    """Lowercase, Arabic-Indic digits -> ASCII, drop thousands separators (5,000 -> 5000)."""
    t = text.translate(_DIGITS).lower()
    return re.sub(r"(?<=\d)[,٬،](?=\d)", "", t)


def contains(answer: str, needle: str) -> bool:
    """Substring match; numeric needles match whole numbers only ('5' does not match '5000')."""
    a, n = normalize_for_match(answer), normalize_for_match(needle)
    if _NUMBER.fullmatch(n):
        return re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", a) is not None
    return n in a


def includes_all(answer: str, needles: tuple[str, ...] | list[str]) -> bool:
    return all(contains(answer, n) for n in needles)


def includes_any(answer: str, needles: tuple[str, ...] | list[str]) -> bool:
    return any(contains(answer, n) for n in needles)