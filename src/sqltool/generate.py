"""LLM SQL generation with one repair retry. The LLM output is never executed directly."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqltool.execute import SqlExecutionError, SqlResult, execute_readonly
from sqltool.format import format_answer, refusal_text
from sqltool.schema import MAX_ROWS
from sqltool.validate import SqlRejected, validate_sql

Complete = Callable[[list[dict[str, str]]], Awaitable[str]]

POLICY_CODES = frozenset({
    "forbidden_table", "forbidden_column", "forbidden_function", "not_select", "comments",
    "multi_statement", "recursive", "placeholder", "denied",
})
MAX_QUESTION_CHARS = 1000


@dataclass
class SqlOutcome:
    status: str                       # answered | refused | clarify
    answer: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    refusal_reason: str | None = None  # no_evidence | policy
    clarification: str | None = None
    llm_calls: int = 0
    failure_codes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def parse_decision(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in output")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    outcome = data.get("outcome") if isinstance(data, dict) else None
    if outcome == "sql":
        if not isinstance(data.get("sql"), str) or not data["sql"].strip():
            raise ValueError("outcome 'sql' needs a non-empty 'sql' string")
    elif outcome == "clarify":
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            raise ValueError("outcome 'clarify' needs a 'question' string")
    elif outcome != "cannot_answer":
        raise ValueError("outcome must be sql, clarify or cannot_answer")
    return data


def _refuse(reason: str, language: str, **kw: Any) -> SqlOutcome:
    return SqlOutcome(status="refused", answer=refusal_text(reason, language), refusal_reason=reason, **kw)


async def answer_sql_question(
    question: str,
    schema_prompt: str,
    db_path: str | Path,
    complete: Complete,
    *,
    language: str = "en",
    max_repairs: int = 1,
    timeout_s: float = 2.0,
) -> SqlOutcome:
    messages = [
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": f"Question:\n{question[:MAX_QUESTION_CHARS]}"},
    ]
    llm_calls, llm_ms, sql_ms = 0, 0.0, 0.0
    codes: list[str] = []

    def timings() -> dict[str, float]:
        return {"llm": round(llm_ms, 1), "sql": round(sql_ms, 1)}

    for attempt in range(max_repairs + 1):
        t0 = time.perf_counter()
        raw = await complete(messages)
        llm_ms += (time.perf_counter() - t0) * 1000
        llm_calls += 1

        code, message = "", ""
        try:
            decision = parse_decision(raw)
        except ValueError as exc:
            code, message = "invalid_output", str(exc)
        else:
            if decision["outcome"] == "cannot_answer":
                return _refuse("no_evidence", language, llm_calls=llm_calls, failure_codes=codes, timings_ms=timings())
            if decision["outcome"] == "clarify":
                return SqlOutcome(status="clarify", answer=decision["question"], clarification=decision["question"],
                                  llm_calls=llm_calls, failure_codes=codes, timings_ms=timings())
            try:
                validated = validate_sql(decision["sql"])
                t1 = time.perf_counter()
                try:
                    result: SqlResult = await asyncio.to_thread(
                        execute_readonly, db_path, validated.sql, max_rows=MAX_ROWS, timeout_s=timeout_s
                    )
                finally:
                    sql_ms += (time.perf_counter() - t1) * 1000
            except SqlRejected as exc:
                code, message = exc.code, exc.message
            except SqlExecutionError as exc:
                if exc.code in ("db_missing", "schema_mismatch"):
                    raise  # infrastructure problem: fail loudly
                code, message = exc.code, exc.message
            else:
                return SqlOutcome(
                    status="answered", answer=format_answer(result, language), sql=validated.sql,
                    columns=result.columns, rows=result.rows, row_count=result.row_count,
                    truncated=result.truncated, llm_calls=llm_calls, failure_codes=codes, timings_ms=timings(),
                )

        codes.append(code)
        if code == "timeout" or attempt == max_repairs:
            reason = "policy" if code in POLICY_CODES else "no_evidence"
            return _refuse(reason, language, llm_calls=llm_calls, failure_codes=codes, timings_ms=timings())
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That output was rejected ({code}): {message}\n"
                "Return corrected JSON using only the allowed tables and columns, or cannot_answer."
            )},
        ]
    raise AssertionError("unreachable")