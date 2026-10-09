"""Read-only SQLite executor: ro URI + query_only + authorizer + progress-handler timeout."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqltool.schema import ALLOWED_SCHEMA, MAX_ROWS, SECRET_MARKER

# SQLite C constants (stable): authorizer return codes and action codes
_OK, _DENY = 0, 1
_READ, _SELECT, _FUNCTION = 20, 21, 31

# Names as SQLite reports them (lower-case). A denied legitimate function is named in the error.
ALLOWED_FUNCTIONS = frozenset({
    "count", "sum", "total", "avg", "min", "max", "round", "abs", "lower", "upper", "length",
    "coalesce", "ifnull", "nullif", "iif", "trim", "ltrim", "rtrim", "substr", "substring",
    "replace", "instr", "typeof", "like", "glob", "likely", "unlikely", "likelihood",
    "date", "datetime", "time", "strftime", "julianday", "group_concat",
})


class SqlExecutionError(RuntimeError):
    """code: db_missing | schema_mismatch | denied | timeout | sql_error"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SqlResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise SqlExecutionError("db_missing", f"database not found: {path} (run src/supportdb/seed.py)")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _make_authorizer(denied: list[str]):
    def authorizer(action: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None) -> int:
        if action == _SELECT:
            return _OK
        if action == _READ:
            table, column = (arg1 or "").lower(), (arg2 or "").lower()
            cols = ALLOWED_SCHEMA.get(table)
            if cols is not None and (column == "" or column in cols):
                return _OK
            denied.append(f"read of {table}.{column}")
            return _DENY
        if action == _FUNCTION:
            if (arg2 or "").lower() in ALLOWED_FUNCTIONS:
                return _OK
            denied.append(f"function {arg2}")
            return _DENY
        denied.append(f"action code {action}")
        return _DENY

    return authorizer


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return value


def execute_readonly(
    db_path: str | Path, sql: str, *, max_rows: int = MAX_ROWS, timeout_s: float = 2.0
) -> SqlResult:
    conn = connect_readonly(db_path)
    denied: list[str] = []
    timed_out = False
    start = time.perf_counter()
    deadline = time.monotonic() + timeout_s

    def progress() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # abort the query
        return 0

    try:
        conn.set_authorizer(_make_authorizer(denied))
        conn.set_progress_handler(progress, 1000)
        try:
            cur = conn.execute(sql)
            fetched = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in (cur.description or [])]
        except sqlite3.Error as exc:
            if denied:
                raise SqlExecutionError("denied", f"not allowed: {denied[0]}") from exc
            if timed_out:
                raise SqlExecutionError("timeout", f"query exceeded {timeout_s:g}s") from exc
            raise SqlExecutionError("sql_error", str(exc)) from exc
    finally:
        conn.close()

    truncated = len(fetched) > max_rows
    rows = [[_jsonable(v) for v in row] for row in fetched[:max_rows]]
    for cell in [*columns, *(v for row in rows for v in row)]:
        if isinstance(cell, str) and SECRET_MARKER in cell:
            raise SqlExecutionError("denied", "result contains a protected value")
    return SqlResult(columns, rows, len(rows), truncated, round((time.perf_counter() - start) * 1000, 2))


def fetch_sample_rows(db_path: str | Path) -> dict[str, dict[str, Any]]:
    """First row per allowed table, allowed columns only (used in the schema prompt)."""
    conn = connect_readonly(db_path)
    out: dict[str, dict[str, Any]] = {}
    try:
        for table, cols in ALLOWED_SCHEMA.items():
            try:
                row = conn.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY 1 LIMIT 1").fetchone()
            except sqlite3.Error as exc:
                raise SqlExecutionError("schema_mismatch", f"{table}: {exc}") from exc
            if row is not None:
                out[table] = dict(zip(cols, row))
    finally:
        conn.close()
    return out