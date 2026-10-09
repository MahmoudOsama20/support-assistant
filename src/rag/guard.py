"""Safety checks on model output and retrieved text. Pure functions, no model calls."""
from __future__ import annotations

import re

from rag.normalize import normalize_text

SYSTEM_CANARY = "CANARY-7c2e91"  # placed in the system prompt; must never appear in output
OFFICIAL_DOMAIN = "nilewallet.example"
_HOST = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b")
_INJECTION = [re.compile(p) for p in (
    r"ignore (?:all |any |the )?(?:previous|prior|above) instructions",
    r"disregard (?:all |any |the )?(?:previous|prior|above)",
    r"system note",
    r"تجاهل (?:كل |جميع )?(?:التعليمات|الاوامر)",  # matched after normalization (alef folded)
)]


def leaks_canary(text: str) -> bool:
    return SYSTEM_CANARY.lower() in normalize_text(text)


def looks_like_injection(text: str) -> bool:
    t = normalize_text(text)
    return any(p.search(t) for p in _INJECTION)


def foreign_host(text: str) -> str | None:
    """First hostname/email domain in the text that is not the official domain or a subdomain of it."""
    for host in _HOST.findall(normalize_text(text)):
        if host != OFFICIAL_DOMAIN and not host.endswith("." + OFFICIAL_DOMAIN):
            return host
    return None


def unsafe_reason(text: str) -> str | None:
    if leaks_canary(text):
        return "prompt_leak"
    if foreign_host(text):
        return "foreign_link"
    if looks_like_injection(text):
        return "injection_text"
    return None