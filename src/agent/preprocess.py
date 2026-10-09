"""Query preprocessing for the agent: clean text, cap length, detect language.

The route model was trained on raw route rows, so we do NOT apply the Arabic
retrieval normalisation here (that is for BM25 only).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.normalize import detect_language  # noqa: E402

MAX_QUERY_CHARS = 1000

# C0/C1-ish control chars (keeping \t \n \r, which collapse to spaces below),
# zero-width chars, bidi marks and bidi overrides/isolates.
_STRIP = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Preprocessed:
    original: str
    text: str
    language: str  # "ar" | "en"
    truncated: bool

    @property
    def is_empty(self) -> bool:
        return not self.text


def preprocess(raw: str, max_chars: int = MAX_QUERY_CHARS) -> Preprocessed:
    if not isinstance(raw, str):
        raise TypeError(f"query must be str, got {type(raw).__name__}")
    text = unicodedata.normalize("NFKC", raw)
    text = _STRIP.sub("", text)
    text = _WS.sub(" ", text).strip()
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip()
    return Preprocessed(original=raw, text=text, language=detect_language(text), truncated=truncated)