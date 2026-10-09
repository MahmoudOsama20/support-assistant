#!/usr/bin/env python3
"""Split KB docs into one chunk per '## ' section.

Run: python src/rag/chunking.py [--docs-dir data/kb/docs]
"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "kb" / "docs"
REQUIRED = ("doc_id", "title", "language", "topic", "version", "effective_date", "status")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    language: str
    topic: str
    status: str
    effective_date: str
    version: str
    injection_test: bool = False

    @property
    def index_text(self) -> str:
        """What retrievers index: title and heading are prefixed to the section body."""
        return f"{self.title}\n{self.heading}\n{self.text}"


def _split_front_matter(text: str, name: str) -> tuple[dict[str, str], str]:
    match = _FRONT.match(text.replace("\r\n", "\n").lstrip("\ufeff"))
    if not match:
        raise ValueError(f"{name}: missing YAML front matter")
    fields: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
    return fields, match.group(2)


def _sections(body: str, title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    heading, lines = title, []  # text before the first '## ' (if any) is filed under the title
    for line in body.split("\n"):
        if line.startswith("## "):
            sections.append((heading, lines))
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    sections.append((heading, lines))
    cleaned = [(h, "\n".join(ls).strip()) for h, ls in sections]
    return [(h, t) for h, t in cleaned if t]


def chunk_doc(path: Path) -> list[Chunk]:
    path = Path(path)
    fields, body = _split_front_matter(path.read_text(encoding="utf-8"), path.name)
    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"{path.name}: missing front matter fields {missing}")
    if fields["doc_id"] != path.stem:
        raise ValueError(f"{path.name}: doc_id {fields['doc_id']!r} does not match filename")
    injected = fields.get("injection_test", "").lower() == "true"
    return [
        Chunk(f"{fields['doc_id']}#c{i}", fields["doc_id"], fields["title"], heading, text,
              fields["language"], fields["topic"], fields["status"], fields["effective_date"],
              fields["version"], injected)
        for i, (heading, text) in enumerate(_sections(body, fields["title"]), start=1)
    ]


def load_chunks(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Chunk]:
    paths = sorted(Path(docs_dir).glob("kb_*.md"))
    if not paths:
        raise FileNotFoundError(f"no kb_*.md files in {docs_dir}")
    return [chunk for path in paths for chunk in chunk_doc(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    args = parser.parse_args()
    chunks = load_chunks(args.docs_dir)
    sizes = sorted((len(c.index_text.split()), c.chunk_id) for c in chunks)
    langs = Counter(c.language for c in chunks)
    print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} docs; by language {dict(langs)}")
    print(f"words per chunk (title+heading+body): min {sizes[0][0]}, "
          f"median {statistics.median(s for s, _ in sizes):.0f}, max {sizes[-1][0]}")
    print("shortest 5:", ", ".join(f"{cid} ({n}w)" for n, cid in sizes[:5]))


if __name__ == "__main__":
    main()