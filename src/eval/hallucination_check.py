#!/usr/bin/env python3
"""Offline grounding check on a results file (no tokens).
Flags answered RAG responses whose numbers are absent from the cited chunks, or that contain forbidden strings.
Prints wrong-route answers separately and a seeded sample of unflagged answers for MANUAL review.

Run: python src/eval/hallucination_check.py results/eval/final-120b.json [--sample 20]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cases import normalize_for_match  # noqa: E402
from eval.scoring import is_rag_category  # noqa: E402

_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(normalize_for_match(text)))


def unsupported_numbers(answer: str, chunk_texts: list[str]) -> list[str]:
    return sorted(numbers(answer) - numbers(" ".join(chunk_texts)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--sample", type=int, default=20)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from rag.chunking import load_chunks

    chunks = {c.chunk_id: c for c in load_chunks()}
    rows = json.loads(args.path.read_text(encoding="utf-8"))["cases"]
    answered = [r for r in rows if r["payload"].get("type") == "rag_answer"]
    flagged, clean = [], []
    for r in answered:
        pl = r["payload"]
        texts = []
        for c in pl.get("citations") or []:
            ch = chunks.get(c.get("chunk_id"))
            if ch:
                texts.append(f"{ch.title} {ch.heading} {ch.text}")
        bad = unsupported_numbers(pl.get("answer") or "", texts)
        (flagged if bad or r["leaked"] else clean).append((r, bad))
    wrong_route = [r for r in rows if is_rag_category(r["category"]) and r["expected_status"] == "success"
                   and r["status"] == "success" and r["payload"].get("type") != "rag_answer"]
    print(f"answered RAG responses: {len(answered)} | flagged: {len(flagged)} "
          f"| rate: {len(flagged) / len(answered) if answered else 'n/a'}")
    for r, bad in flagged:
        print(f"FLAG {r['id']}: unsupported numbers {bad}, leaked {r['leaked']}\n   {r['payload'].get('answer')!r}")
    print(f"\nwrong-route answers (KB question answered by SQL): {[r['id'] for r in wrong_route]}")
    rng = random.Random(42)
    sample = rng.sample(clean, min(args.sample, len(clean)))
    print(f"\n-- manual review sample ({len(sample)} unflagged; check each against fact_sheet.md) --")
    for r, _ in sample:
        print(f"{r['id']} [{r['language']}] {r['payload'].get('answer')!r}")


if __name__ == "__main__":
    main()