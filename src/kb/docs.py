#!/usr/bin/env python
"""Load and validate the synthetic Nile Wallet knowledge base (data/kb/docs/*.md)."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = ROOT / "data" / "kb" / "docs"

REQUIRED_FIELDS = ("doc_id", "title", "language", "category", "topic",
                   "version", "effective_date", "status", "source")
LANGUAGES = {"ar", "en"}
STATUSES = {"active", "superseded"}
WORDS_HARD = (120, 450)     # outside -> error
WORDS_TARGET = (180, 350)   # outside -> warning
INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"ignore (all )?(previous|prior|above) instructions",
    r"system (note|prompt|override)",
    r"you are now",
    "تجاهل (جميع )?التعليمات",
)]
_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


class KBError(Exception):
    pass


@dataclass(frozen=True)
class Doc:
    meta: dict
    body: str
    path: Path

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    @property
    def n_sections(self) -> int:
        return len(re.findall(r"^## .+$", self.body, flags=re.MULTILINE))


def parse_doc(path: Path) -> Doc:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").lstrip("\ufeff")
    match = _FRONT.match(text)
    if not match:
        raise KBError(f"{path.name}: missing '---' front matter")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise KBError(f"{path.name}: invalid YAML front matter: {exc}") from exc
    if not isinstance(meta, dict):
        raise KBError(f"{path.name}: front matter is not a mapping")
    return Doc(meta=meta, body=match.group(2).strip(), path=path)


def load_docs(docs_dir: Path) -> list[Doc]:
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise KBError(f"no .md files in {docs_dir}")
    return [parse_doc(p) for p in paths]


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0600" <= c <= "\u06FF" for c in letters) / len(letters)


def validate_docs(docs: list[Doc]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for d in docs:
        name, m = d.path.name, d.meta
        missing = [f for f in REQUIRED_FIELDS if f not in m]
        if missing:
            errors.append(f"{name}: missing fields {missing}")
            continue
        if m["doc_id"] != d.path.stem:
            errors.append(f"{name}: doc_id '{m['doc_id']}' does not match filename")
        if m["doc_id"] in seen:
            errors.append(f"{name}: duplicate doc_id '{m['doc_id']}'")
        seen.add(m["doc_id"])
        if m["language"] not in LANGUAGES:
            errors.append(f"{name}: language must be one of {sorted(LANGUAGES)}")
        if m["status"] not in STATUSES:
            errors.append(f"{name}: status must be one of {sorted(STATUSES)}")
        try:
            date.fromisoformat(str(m["effective_date"]))
        except ValueError:
            errors.append(f"{name}: effective_date '{m['effective_date']}' is not YYYY-MM-DD")
        ratio = arabic_ratio(d.body)
        if m["language"] == "ar" and ratio < 0.6:
            errors.append(f"{name}: language=ar but only {ratio:.0%} of letters are Arabic")
        if m["language"] == "en" and ratio > 0.02:
            errors.append(f"{name}: language=en but {ratio:.0%} of letters are Arabic")
        if d.n_sections < 2:
            errors.append(f"{name}: needs at least 2 '## ' sections (chunking is by section)")
        wc = d.word_count
        if not WORDS_HARD[0] <= wc <= WORDS_HARD[1]:
            errors.append(f"{name}: {wc} words, outside hard range {WORDS_HARD}")
        elif not WORDS_TARGET[0] <= wc <= WORDS_TARGET[1]:
            warnings.append(f"{name}: {wc} words, outside target range {WORDS_TARGET}")
        flagged = any(p.search(d.body) for p in INJECTION_PATTERNS)
        declared = bool(m.get("injection_test", False))
        if flagged and not declared:
            errors.append(f"{name}: matches an injection pattern but has no 'injection_test: true'")
        if declared and not flagged:
            errors.append(f"{name}: declares injection_test but no injection pattern matched")
    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ap.add_argument("--expect-count", type=int, default=None)
    args = ap.parse_args()
    try:
        docs = load_docs(args.docs_dir)
    except KBError as exc:
        sys.exit(f"ERROR: {exc}")
    errors, warnings = validate_docs(docs)
    if args.expect_count is not None and len(docs) != args.expect_count:
        errors.append(f"expected {args.expect_count} docs, found {len(docs)}")

    print(f"{'doc_id':<8} {'lang':<4} {'words':>5} {'sec':>3} {'status':<10} topic")
    for d in docs:
        m = d.meta
        print(f"{str(m.get('doc_id')):<8} {str(m.get('language')):<4} {d.word_count:>5} "
              f"{d.n_sections:>3} {str(m.get('status')):<10} {m.get('topic')}")
    by_lang: dict[str, int] = {}
    for d in docs:
        lang = str(d.meta.get("language"))
        by_lang[lang] = by_lang.get(lang, 0) + 1
    print(f"\n{len(docs)} docs, by language: {by_lang}")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()