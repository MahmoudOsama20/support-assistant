"""Async OpenAI-compatible chat client: env config, on-disk cache, 429/5xx backoff honouring Retry-After."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_cache"
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any LLM failure."""


class LLMUnavailable(LLMError):
    """Rate-limited, overloaded or unreachable: the caller should degrade."""


class LLMTruncated(LLMError):
    """The completion was empty because the token budget ran out (reasoning tokens count)."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 1500
    reasoning_effort: str | None = "low"
    temperature: float = 0.0
    timeout_s: float = 60.0
    max_retries: int = 3
    max_wait_s: float = 30.0
    min_interval_s: float = 0.0
    max_concurrency: int = 2

    @classmethod
    def from_env(cls) -> LLMSettings:
        env = os.environ.get
        effort = env("LLM_REASONING_EFFORT", "low").strip().lower()
        return cls(
            base_url=env("LLM_BASE_URL", cls.base_url).rstrip("/"),
            api_key=env("LLM_API_KEY") or env("GROQ_API_KEY") or "",
            model=env("LLM_MODEL", cls.model),
            max_tokens=int(env("LLM_MAX_TOKENS", cls.max_tokens)),
            reasoning_effort=None if effort in ("", "none") else effort,
            max_wait_s=float(env("LLM_MAX_WAIT_S", cls.max_wait_s)),
            min_interval_s=float(env("LLM_MIN_INTERVAL_S", cls.min_interval_s)),
        )


@dataclass(frozen=True)
class LLMReply:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float  # successful attempt only (excludes backoff sleeps)
    retries: int = 0
    cached: bool = False


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


class LLMClient:
    def __init__(self, settings: LLMSettings, cache_dir: Path | None = None, *,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep) -> None:
        if not settings.api_key:
            raise LLMError("no API key: set LLM_API_KEY or GROQ_API_KEY")
        self.settings, self.cache_dir, self._sleep = settings, cache_dir, sleep
        self._http = httpx.AsyncClient(base_url=settings.base_url, timeout=settings.timeout_s,
                                       headers={"Authorization": f"Bearer {settings.api_key}"},
                                       transport=transport)
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._pace_lock = asyncio.Lock()
        self._last_call = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    def _body(self, system: str, user: str, max_tokens: int | None) -> dict:
        s = self.settings
        body = {"model": s.model, "temperature": s.temperature, "max_tokens": max_tokens or s.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if s.reasoning_effort:
            body["reasoning_effort"] = s.reasoning_effort
        return body

    async def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> LLMReply:
        body = self._body(system, user, max_tokens)
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        reply = await self._request(body)
        self._cache_write(key, reply)
        return reply

    # ---- internals ----
    def _cache_read(self, key: str) -> LLMReply | None:
        if self.cache_dir is None:
            return None
        try:
            data = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
            return LLMReply(**{**data, "cached": True})
        except (OSError, ValueError, TypeError):
            return None

    def _cache_write(self, key: str, reply: LLMReply) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(reply).items() if k != "cached"}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def _pace(self) -> None:
        if self.settings.min_interval_s <= 0:
            return
        async with self._pace_lock:
            wait = self._last_call + self.settings.min_interval_s - time.monotonic()
            if wait > 0:
                await self._sleep(wait)
            self._last_call = time.monotonic()

    async def _request(self, body: dict) -> LLMReply:
        s = self.settings
        attempts = s.max_retries + 1
        last = "no response"
        for attempt in range(attempts):
            resp: httpx.Response | None = None
            async with self._sem:
                await self._pace()
                started = time.perf_counter()
                try:
                    resp = await self._http.post("chat/completions", json=body)
                except httpx.TransportError as e:
                    last = f"{type(e).__name__}: {e}"
                latency_ms = (time.perf_counter() - started) * 1000
            if resp is not None and resp.status_code == 200:
                return self._parse(resp, latency_ms, retries=attempt)
            if resp is not None:
                if resp.status_code not in RETRYABLE:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            wait = _retry_after(resp) if resp is not None else None
            if wait is None:
                wait = 2.0 ** attempt
            if wait > s.max_wait_s:
                raise LLMUnavailable(f"server asked to wait {wait:.0f}s (max_wait_s={s.max_wait_s:.0f}); {last}")
            if attempt == attempts - 1:
                raise LLMUnavailable(f"gave up after {attempts} attempts; {last}")
            await self._sleep(wait)
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(resp: httpx.Response, latency_ms: float, retries: int) -> LLMReply:
        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {resp.text[:200]}") from e
        if not text.strip():
            msg = f"empty completion (finish_reason={finish}); raise LLM_MAX_TOKENS if reasoning used the budget"
            raise (LLMTruncated if finish == "length" else LLMError)(msg)
        return LLMReply(text, finish, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        latency_ms, retries)