"""Deterministic pre-routing rules."""
from __future__ import annotations

MIN_WORDS = 3


def is_too_short(text: str, min_words: int = MIN_WORDS) -> bool:
    """A query under `min_words` words with no digit (no ticket/customer id) carries too little to route."""
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return len(words) < min_words and not any(c.isdigit() for c in text)