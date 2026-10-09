from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from llm.client import LLMClient, LLMError, LLMSettings, LLMTruncated, LLMUnavailable

SLEEPS: list[float] = []


@pytest.fixture(autouse=True)
def _clear_sleeps() -> None:
    SLEEPS.clear()


def _ok(content: str = "hello", finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def _run(handler, cache_dir: Path | None = None, calls: int = 1, **overrides):
    async def fake_sleep(seconds: float) -> None:
        SLEEPS.append(seconds)

    async def main():
        settings = LLMSettings(api_key="k", base_url="http://test/v1", **overrides)
        client = LLMClient(settings, cache_dir=cache_dir, transport=httpx.MockTransport(handler), sleep=fake_sleep)
        try:
            return [await client.complete("sys", "usr") for _ in range(calls)]
        finally:
            await client.aclose()

    return asyncio.run(main())


def test_success_and_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()

    (reply,) = _run(handler)
    assert (reply.text, reply.prompt_tokens, reply.completion_tokens) == ("hello", 10, 5)
    assert reply.retries == 0 and not reply.cached
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://test/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert body["model"] == "openai/gpt-oss-120b" and body["temperature"] == 0.0
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 1500
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_retry_after_is_honoured() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "2"}) if len(calls) == 1 else _ok()

    (reply,) = _run(handler)
    assert reply.retries == 1 and SLEEPS == [2.0] and len(calls) == 2


def test_long_retry_after_fails_fast() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "3600"}, text="daily token limit")

    with pytest.raises(LLMUnavailable, match="3600"):
        _run(handler)
    assert SLEEPS == [] and len(calls) == 1


def test_server_errors_exhaust_retries() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(LLMUnavailable, match="gave up"):
        _run(handler, max_retries=2)
    assert len(calls) == 3 and SLEEPS == [1.0, 2.0]


def test_client_error_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad parameter")

    with pytest.raises(LLMError) as exc:
        _run(handler)
    assert not isinstance(exc.value, LLMUnavailable) and len(calls) == 1


def test_truncated_reasoning_is_not_cached(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _ok(content="", finish="length")

    for _ in range(2):
        with pytest.raises(LLMTruncated):
            _run(handler, cache_dir=tmp_path)
    assert len(calls) == 2


def test_cache_hit(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _ok()

    first, second = _run(handler, cache_dir=tmp_path, calls=2)
    assert len(calls) == 1 and not first.cached and second.cached and first.text == second.text


def test_reasoning_effort_omitted_when_none() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok()

    _run(handler, reasoning_effort=None)
    assert "reasoning_effort" not in seen[0]


def test_missing_api_key_fails_loudly() -> None:
    with pytest.raises(LLMError, match="API key"):
        LLMClient(LLMSettings(api_key=""))


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MAX_TOKENS", "LLM_MAX_WAIT_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "12")
    s = LLMSettings.from_env()
    assert (s.api_key, s.model, s.reasoning_effort, s.min_interval_s) == ("g", "m", None, 12.0)
    assert s.base_url == "https://api.groq.com/openai/v1" and s.max_tokens == 1500