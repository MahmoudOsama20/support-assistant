from __future__ import annotations

import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "kb" / "docs"
CANARY = "LOYALTY-OVERRIDE-4471"


def _read(doc_id: str) -> tuple[dict[str, str], str]:
    text = (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    _, front, body = text.split("---\n", 2)
    fields = {k: v.strip().strip('"') for k, v in re.findall(r"^(\w+):\s*(.+)$", front, re.M)}
    return fields, body


def _all_ids() -> list[str]:
    return sorted(p.stem for p in DOCS_DIR.glob("kb_*.md"))


def test_total_is_32_docs() -> None:
    assert len(_all_ids()) == 32


def test_single_language_docs() -> None:
    for i in range(25, 29):
        assert _read(f"kb_{i:03d}")[0]["language"] == "ar"
    for i in range(29, 33):
        assert _read(f"kb_{i:03d}")[0]["language"] == "en"


def test_only_kb_031_is_superseded() -> None:
    superseded = [d for d in _all_ids() if _read(d)[0]["status"] != "active"]
    assert superseded == ["kb_031"]
    fields, body = _read("kb_031")
    assert fields["status"] == "superseded"
    assert "2025-12-31" in body and "2026-01-01" in body


def test_only_kb_032_is_injected() -> None:
    flagged = [d for d in _all_ids() if _read(d)[0].get("injection_test") == "true"]
    assert flagged == ["kb_032"]
    carriers = [d for d in _all_ids() if CANARY in _read(d)[1]]
    assert carriers == ["kb_032"]


def test_topics_are_unique_per_language() -> None:
    seen: set[tuple[str, str]] = set()
    for d in _all_ids():
        fields, _ = _read(d)
        key = (fields["topic"], fields["language"])
        assert key not in seen, f"duplicate {key}"
        seen.add(key)