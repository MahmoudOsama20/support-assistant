"""Deterministic result text (no LLM)."""
from __future__ import annotations

from typing import Any

from sqltool.execute import SqlResult

PREVIEW_ROWS = 5

_T = {
    "en": {
        "empty": "No matching records found.",
        "rows": "{n} rows returned.",
        "trunc": " Results truncated to the first {n}.",
        "refused": "I can't answer that from the data I'm allowed to query.",
        "unsafe": "That request isn't permitted.",
    },
    "ar": {
        "empty": "لم يتم العثور على سجلات مطابقة.",
        "rows": "عدد الصفوف: {n}.",
        "trunc": " تم اقتطاع النتائج إلى أول {n}.",
        "refused": "لا يمكنني الإجابة عن ذلك من البيانات المسموح لي بالاستعلام عنها.",
        "unsafe": "هذا الطلب غير مسموح به.",
    },
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{round(value, 2):.2f}".rstrip("0").rstrip(".")
    return str(value)


def _lang(language: str) -> dict[str, str]:
    return _T["ar" if language == "ar" else "en"]


def refusal_text(reason: str, language: str) -> str:
    return _lang(language)["unsafe" if reason == "policy" else "refused"]


def format_answer(result: SqlResult, language: str) -> str:
    t = _lang(language)
    if result.row_count == 0:
        return t["empty"]
    pairs = lambda row: ", ".join(f"{c}: {_fmt(v)}" for c, v in zip(result.columns, row))  # noqa: E731
    if result.row_count == 1:
        return pairs(result.rows[0])
    head = t["rows"].format(n=result.row_count)
    if result.truncated:
        head += t["trunc"].format(n=result.row_count)
    lines = [pairs(r) for r in result.rows[:PREVIEW_ROWS]]
    if result.row_count > PREVIEW_ROWS:
        lines.append("...")
    return head + "\n" + "\n".join(lines)