#!/usr/bin/env python3
"""BM25 over KB chunks with Arabic-aware tokenization.

Run: python src/rag/bm25.py "your query" [-k 5] [--no-stem]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

if __package__ in (None, ""):  # direct execution: make `rag` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi  # noqa: E402

from rag.chunking import Chunk, load_chunks  # noqa: E402
from rag.normalize import tokenize  # noqa: E402


class Hit(NamedTuple):
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], *, light_stem: bool = True) -> None:
        if not chunks:
            raise ValueError("cannot build a BM25 index from zero chunks")
        self.chunks = list(chunks)
        self.light_stem = light_stem
        self._bm25 = BM25Okapi([tokenize(c.index_text, light_stem=light_stem) for c in self.chunks])

    def search(self, query: str, k: int = 20) -> list[Hit]:
        """Top-k chunks with score > 0, best first; ties broken by corpus order (deterministic)."""
        tokens = tokenize(query, light_stem=self.light_stem)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-stem", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    hits = BM25Index(load_chunks(), light_stem=not args.no_stem).search(args.query, args.k)
    if not hits:
        print("no lexical match")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(f"{rank}. {hit.score:6.2f}  {c.chunk_id:<10} {c.language} {c.status:<10} "
              f"{c.heading} | {c.text[:70].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()