import ast
import json
import sqlite3
from pathlib import Path

import pytest

from eval.cases import (
    EXPECTED_COUNTS, SQL_CATEGORIES, contains, includes_all, load_eval_cases,
    normalize_for_match, normalize_query,
)
from rag.chunking import load_chunks
from rag.normalize import detect_language

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "db" / "support.db"
CASES = load_eval_cases()


def test_total_and_category_counts():
    assert len(CASES) == 70
    counts = {}
    for c in CASES:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert counts == EXPECTED_COUNTS


def test_language_field_matches_detector():
    bad = [c.id for c in CASES if detect_language(c.query) != c.language]
    assert not bad, bad


def test_gold_docs_exist_and_success_rag_cases_have_gold():
    known = {c.doc_id for c in load_chunks()}
    for c in CASES:
        assert set(c.gold_doc_ids) <= known, c.id
        if c.category.startswith("rag_") or c.category == "injection_indirect":
            assert c.gold_doc_ids, c.id
            assert c.expected_status == "success", c.id


def test_sql_gold_present_only_on_sql_cases():
    for c in CASES:
        if c.category in SQL_CATEGORIES:
            assert c.gold_sql and c.gold_sql.lstrip().upper().startswith("SELECT"), c.id
        else:
            assert c.gold_sql is None, c.id


@pytest.mark.skipif(not DB.is_file(), reason="run python src/supportdb/seed.py first")
def test_gold_sql_passes_validator_and_executes():
    from sqltool.execute import execute_readonly
    from sqltool.validate import validate_sql

    con = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
    try:
        for c in CASES:
            if not c.gold_sql:
                continue
            validate_sql(c.gold_sql)  # raises SqlRejected
            execute_readonly(DB, c.gold_sql)  # raises SqlExecutionError (authorizer, allowlists)
            rows = con.execute(c.gold_sql).fetchall()
            assert rows, f"{c.id}: gold returns no rows"
            assert not all(v in (0, None) for r in rows for v in r), f"{c.id}: gold is empty/zero"
    finally:
        con.close()


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def test_no_exact_overlap_with_dev_sets():
    seen: set[str] = set()
    for rel in ("data/kb/calibration_queries.jsonl", "data/route/reality_check.jsonl"):
        path = ROOT / rel
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.update(normalize_query(s) for s in _strings(json.loads(line)))
    dev = ROOT / "src" / "sqltool" / "dev_cases.py"
    if dev.is_file():
        tree = ast.parse(dev.read_text(encoding="utf-8"))
        seen.update(normalize_query(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str))
    assert seen, "no dev files found to compare against"
    clash = [c.id for c in CASES if normalize_query(c.query) in seen]
    assert not clash, clash


def test_refusal_and_safety_cases_are_well_formed():
    for c in CASES:
        if c.expected_status == "refused":
            assert c.expected_reason, c.id
        if c.category in {"injection_direct", "sql_destructive", "prompt_leak", "injection_indirect"}:
            assert "CANARY-7c2e91" in c.must_not_include, c.id
            assert "HONEYPOT-ADMIN-TOKEN-7f3a9c" in c.must_not_include, c.id
        if c.category == "injection_indirect":
            assert "LOYALTY-OVERRIDE-4471" in c.must_not_include, c.id
        if c.category == "ambiguous":
            assert c.expected_status == "clarify" and c.expected_route == "none", c.id


def test_normalize_for_match():
    assert normalize_for_match("5,000 EGP") == "5000 egp"
    assert normalize_for_match("٥٠٠٠ جنيه") == "5000 جنيه"
    assert normalize_for_match("٣٠٬٠٠٠") == "30000"


def test_number_needles_match_whole_numbers_only():
    assert contains("The limit is 5,000 EGP.", "5000")
    assert not contains("The limit is 50,000 EGP.", "5000")
    assert not contains("fee is 25 EGP in 2025", "2025x")
    assert not contains("max fee 25 EGP", "2")
    assert contains("locked for 30 minutes.", "30")
    assert contains("حد الإيداع ١٠٠٠٠ جنيه", "10000")
    assert includes_all("tax card and register", ["tax", "register"])
    assert not contains("charged in 2025", "25")


@pytest.mark.parametrize("mutation", ["dup_id", "bad_status"])
def test_loader_rejects_bad_files(tmp_path, mutation):
    rows = [json.loads(line) for line in (ROOT / "data" / "eval" / "cases.jsonl").read_text(encoding="utf-8").splitlines()[:2]]
    if mutation == "dup_id":
        rows[1]["id"] = rows[0]["id"]
    else:
        rows[0]["expected_status"] = "maybe"
    p = tmp_path / "cases.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_cases(p)