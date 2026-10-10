"""Adapter between the SQL tool's `complete(messages) -> str` and src/llm/client.py.

Client API (verified): LLMClient.complete(system, user) -> LLMReply(text, ...).
The client has no temperature argument here; determinism comes from its on-disk cache.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from llm.client import LLMClient, LLMSettings
from llm.env import ENV_FILE, load_dotenv_file  # noqa: F401  (re-exported for older imports)

Complete = Callable[[list[dict[str, str]]], Awaitable[str]]


def _flatten(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return system, turns[0]["content"]
    # repair round: keep the original question, the rejected output and the rejection reason
    user = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns)
    return system, user


def make_complete_from(client: LLMClient) -> Complete:
    """Wrap a client someone else owns (no aclose: the owner closes it)."""

    async def complete(messages: list[dict[str, str]]) -> str:
        system, user = _flatten(messages)
        reply = await client.complete(system, user)
        return reply.text

    return complete


def make_complete() -> Complete:
    """Standalone version for the SQL CLI: loads .env, owns its (uncached) client."""
    load_dotenv_file()
    client = LLMClient(LLMSettings.from_env())
    complete = make_complete_from(client)
    complete.aclose = client.aclose  # type: ignore[attr-defined]
    return complete