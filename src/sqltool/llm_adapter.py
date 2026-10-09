"""The only file that touches src/llm/client.py.

Client API (verified): LLMClient(settings).complete(system, user) -> LLMReply; settings via LLMSettings.from_env().
UNVERIFIED: the reply's text attribute is assumed to be `.text`; a wrong guess fails loudly below.
Note: the client has no temperature argument here, so determinism comes from its on-disk cache.
"""
from __future__ import annotations

import dataclasses
from typing import Awaitable, Callable

from llm.client import LLMClient, LLMSettings


import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv_file(path: Path = ENV_FILE) -> None:
    """Minimal .env loader: KEY=VALUE lines; real environment variables take precedence."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def _flatten(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return system, turns[0]["content"]
    # repair round: keep the original question, the rejected output and the rejection reason
    user = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns)
    return system, user


def make_complete() -> Callable[[list[dict[str, str]]], Awaitable[str]]:
    load_dotenv_file()
    client = LLMClient(LLMSettings.from_env())

    async def complete(messages: list[dict[str, str]]) -> str:
        system, user = _flatten(messages)
        reply = await client.complete(system, user)
        text = getattr(reply, "text", None)
        if not isinstance(text, str):
            fields = [f.name for f in dataclasses.fields(reply)] if dataclasses.is_dataclass(reply) else dir(reply)
            raise AttributeError(f"LLMReply has no str .text; available: {fields}")
        return text

    complete.aclose = client.aclose  # type: ignore[attr-defined]
    return complete