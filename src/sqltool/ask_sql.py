#!/usr/bin/env python
"""SQL tool CLI.

  python src/sqltool/ask_sql.py --show-prompt
  python src/sqltool/ask_sql.py --sql "SELECT ..."          # validator + executor only, no LLM
  python src/sqltool/ask_sql.py "question"                  # full tool (needs LLM env vars)
  python src/sqltool/ask_sql.py --dev-cases [--limit N]     # execution match vs gold SQL
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.normalize import detect_language  # noqa: E402
from sqltool.dev_cases import DEV_CASES  # noqa: E402
from sqltool.execute import SqlExecutionError, execute_readonly, fetch_sample_rows  # noqa: E402
from sqltool.generate import answer_sql_question  # noqa: E402
from sqltool.schema import build_schema_prompt  # noqa: E402
from sqltool.validate import SqlRejected, validate_sql  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "db" / "support.db"


def _norm(rows: list[list]) -> list[tuple]:
    return sorted((tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows), key=repr)


def _run_raw(sql: str, db: Path, timeout: float) -> int:
    try:
        validated = validate_sql(sql)
        print("sql:", validated.sql)
        result = execute_readonly(db, validated.sql, timeout_s=timeout)
    except SqlRejected as exc:
        print(f"REJECTED [{exc.code}]: {exc.message}")
        return 2
    except SqlExecutionError as exc:
        print(f"EXECUTION ERROR [{exc.code}]: {exc.message}")
        return 2
    print("columns:", result.columns)
    print("rows:", result.rows)
    print(f"row_count={result.row_count} truncated={result.truncated} elapsed_ms={result.elapsed_ms}")
    return 0


async def _run_question(question: str, db: Path, timeout: float) -> int:
    from sqltool.llm_adapter import make_complete

    prompt = build_schema_prompt(fetch_sample_rows(db))
    complete = make_complete()
    try:
        lang = "ar" if detect_language(question) == "ar" else "en"
        outcome = await answer_sql_question(question, prompt, db, complete, language=lang, timeout_s=timeout)
    finally:
        await complete.aclose()
    print(json.dumps(dataclasses.asdict(outcome), ensure_ascii=False, indent=2))
    return 0


async def _run_dev_cases(db: Path, timeout: float, limit: int | None) -> int:
    from sqltool.llm_adapter import make_complete

    prompt = build_schema_prompt(fetch_sample_rows(db))
    complete = make_complete()
    cases = DEV_CASES[:limit] if limit else DEV_CASES
    records, passed = [], 0
    for case in cases:
        outcome = await answer_sql_question(case.question, prompt, db, complete, language=case.language, timeout_s=timeout)
        if case.gold_sql is None:
            ok, want = outcome.status != "answered", None
        else:
            want = execute_readonly(db, validate_sql(case.gold_sql).sql).rows
            ok = outcome.status == "answered" and _norm(outcome.rows) == _norm(want)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {case.id} [{case.language}] status={outcome.status} calls={outcome.llm_calls} "
              f"codes={outcome.failure_codes}")
        if not ok:
            print(f"     q: {case.question}\n     sql: {outcome.sql}\n     got: {outcome.rows}\n     want: {want}")
        records.append({"id": case.id, "pass": ok, "status": outcome.status, "sql": outcome.sql,
                        "rows": outcome.rows, "failure_codes": outcome.failure_codes, "llm_calls": outcome.llm_calls})
    print(f"\n{passed}/{len(cases)} passed")
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b").replace("/", "_")
    out = ROOT / "results" / "sql" / f"dev_cases_{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "passed": passed, "total": len(cases), "cases": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--sql")
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--dev-cases", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--timeout", type=float, default=2.0)
    a = p.parse_args()

    if a.show_prompt:
        print(build_schema_prompt(fetch_sample_rows(a.db)))
        return 0
    if a.sql:
        return _run_raw(a.sql, a.db, a.timeout)
    if a.dev_cases:
        return asyncio.run(_run_dev_cases(a.db, a.timeout, a.limit))
    if a.question:
        return asyncio.run(_run_question(a.question, a.db, a.timeout))
    p.error("give a question, --sql, --dev-cases or --show-prompt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())