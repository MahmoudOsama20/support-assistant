"""Text normalization shared by BM25, preprocessing and citation verification."""
from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile("[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_DIGITS = {ord(c): str(i % 10) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord("٬"): ",", ord("٫"): "."})  # Arabic thousands / decimal separators
_ALEF = {ord(c): "ا" for c in "أإآٱ"}
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # 5,000 -> 5000 (1,5 untouched)
_TOKEN = re.compile(r"\d+(?:\.\d+)*|\w+")
_ARABIC = re.compile("[\u0600-\u06FF]")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")  # longest first
_MIN_STEM = 2


def normalize_text(text: str, *, fold_ta_marbuta: bool = True) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = s.translate(_DIGITS)
    s = _TASHKEEL.sub("", s).replace(_TATWEEL, "")
    s = s.translate(_ALEF).replace("ى", "ي")
    if fold_ta_marbuta:
        s = s.replace("ة", "ه")
    s = _THOUSANDS.sub("", s)
    return " ".join(s.lower().split())


def _stem(token: str) -> str:
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= _MIN_STEM:
            return token[len(prefix):]
    return token


def tokenize(text: str, *, light_stem: bool = True, fold_ta_marbuta: bool = True) -> list[str]:
    tokens = _TOKEN.findall(normalize_text(text, fold_ta_marbuta=fold_ta_marbuta))
    if light_stem:
        tokens = [_stem(t) if _ARABIC.search(t) else t for t in tokens]
    return tokens