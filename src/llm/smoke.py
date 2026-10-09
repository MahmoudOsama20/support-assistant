#!/usr/bin/env python3
"""Probe the LLM endpoint: parameter acceptance, reasoning-token cost, JSON and Arabic."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import LLMClient, LLMError, LLMSettings  # noqa: E402


SYSTEM = "You are a terse assistant."
PROBES = [
    ("effort_low", {"reasoning_effort": "low", "max_tokens": 300}, "Reply with the single word OK."),
    ("effort_default", {"reasoning_effort": None, "max_tokens": 600}, "Reply with the single word OK."),
    ("tiny_budget", {"reasoning_effort": "low", "max_tokens": 24}, "Reply with the single word OK."),
    ("json_only", {"reasoning_effort": "low", "max_tokens": 300},
     'Return only this JSON object, no other text: {"ok": true, "n": 3}'),
    ("arabic", {"reasoning_effort": "low", "max_tokens": 300}, "أجب بكلمة واحدة فقط: ما هي عاصمة مصر؟"),
]


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = LLMSettings.from_env()
    print(f"model={base.model} base_url={base.base_url}")
    total = 0
    for name, overrides, prompt in PROBES:
        client = LLMClient(replace(base, **overrides))  # no cache: we want real numbers
        try:
            reply = await client.complete(SYSTEM, prompt)
            used = (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
            total += used
            print(f"{name:<15} ok   finish={reply.finish_reason} prompt={reply.prompt_tokens} "
                  f"completion={reply.completion_tokens} {reply.latency_ms:7.0f} ms retries={reply.retries} "
                  f"text={reply.text.strip()[:60]!r}")
        except LLMError as e:
            print(f"{name:<15} FAIL {type(e).__name__}: {str(e)[:200]}")
        finally:
            await client.aclose()
    print(f"tokens used by this smoke test: {total}")


if __name__ == "__main__":
    asyncio.run(main())