from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "kb" / "docs"
IGNORED = {"1", "2"}  # Arabic often writes these as words (يوم واحد, يومان)
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUMBER_RE = re.compile(r"\d[\d,]*\d|\d")


def _load() -> dict[str, list[tuple[str, set[str], str]]]:
    """topic -> [(language, numbers_in_body, doc_id)]"""
    topics: dict[str, list[tuple[str, set[str], str]]] = defaultdict(list)
    for path in sorted(DOCS_DIR.glob("kb_*.md")):
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        _, front, body = text.split("---\n", 2)
        topic = re.search(r"^topic:\s*(\S+)", front, re.M).group(1)
        lang = re.search(r"^language:\s*(\S+)", front, re.M).group(1)
        nums = {n.replace(",", "") for n in NUMBER_RE.findall(body.translate(ARABIC_INDIC))}
        topics[topic].append((lang, nums - IGNORED, path.stem))
    return topics


def test_one_doc_per_language_per_topic() -> None:
    for topic, docs in _load().items():
        langs = [lang for lang, _, _ in docs]
        assert len(langs) == len(set(langs)), f"{topic}: duplicate language {langs}"


def test_bilingual_pairs_share_the_same_numbers() -> None:
    for topic, docs in _load().items():
        by_lang = {lang: (nums, doc_id) for lang, nums, doc_id in docs}
        if {"en", "ar"} <= by_lang.keys():
            en, ar = by_lang["en"], by_lang["ar"]
            assert en[0] == ar[0], (
                f"{topic}: {en[1]} vs {ar[1]} differ: "
                f"en-only={sorted(en[0] - ar[0])} ar-only={sorted(ar[0] - en[0])}"
            )