"""Support agent: a plain async state machine.

preprocess -> (empty?) -> injection flag -> route + intent models -> decide() -> RAG | SQL | CLARIFY | REFUSE.
Tools are injected as `async (query, language) -> outcome`; the LLM never picks the tool.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.messages import text_for  # noqa: E402
from agent.policy import DEFAULT_TAU_ROUTE, decide  # noqa: E402
from agent.preprocess import preprocess  # noqa: E402
from agent.trace import Trace  # noqa: E402
from llm.client import LLMError, LLMUnavailable  # noqa: E402
from rag.guard import leaks_canary, looks_like_injection  # noqa: E402

logger = logging.getLogger("agent")

Tool = Callable[[str, str], Awaitable[Any]]


class RouteModel(Protocol):
    labels: Sequence[str]

    def predict(self, text: str) -> list[float]: ...


class IntentModel(Protocol):
    def predict(self, text: str) -> tuple[str, float]: ...


@dataclass(frozen=True)
class AgentConfig:
    tau_route: float = DEFAULT_TAU_ROUTE
    tool_timeout_s: float = 45.0
    model_concurrency: int = 2
    zero_rows_refuse: bool = True  # SQL with 0 rows -> refuse no_evidence


@dataclass
class AgentResult:
    status: str                      # answered | refused | clarify | error
    action: str                      # rag | sql | clarify | refuse
    language: str
    route: str                       # top-1 route label ("none" for an empty query)
    route_confidence: float
    answer: str = ""
    tool: str | None = None          # rag | sql
    refusal_reason: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None
    citations: list[Any] = field(default_factory=list)
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[Any] = field(default_factory=list)
    row_count: int | None = None
    truncated: bool = False
    degraded: bool = False
    error_code: str | None = None
    retryable: bool = False
    flags: dict[str, bool] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


class SupportAgent:
    def __init__(self, route_model: RouteModel, intent_model: IntentModel | None,
                 rag_tool: Tool, sql_tool: Tool, config: AgentConfig = AgentConfig()) -> None:
        self.route_model, self.intent_model = route_model, intent_model
        self.rag_tool, self.sql_tool, self.config = rag_tool, sql_tool, config
        self._sem = asyncio.Semaphore(config.model_concurrency)

    async def _infer(self, fn: Callable[..., Any], *args: Any) -> Any:
        async with self._sem:
            return await asyncio.to_thread(fn, *args)

    async def _none(self) -> None:
        return None

    async def handle(self, raw_query: str) -> AgentResult:
        trace = Trace()
        with trace.span("preprocess"):
            pre = preprocess(raw_query)
        lang = pre.language
        flags = {"injection_flagged": False, "truncated": pre.truncated}
        base: dict[str, Any] = {"language": lang, "route": "none", "route_confidence": 0.0, "flags": flags}

        def make(**kw: Any) -> AgentResult:
            timings = dict(trace.timings_ms)
            timings["total"] = trace.total_ms()
            return AgentResult(**{**base, "timings_ms": timings, **kw})

        if pre.is_empty:
            return make(status="clarify", action="clarify", answer=text_for("clarify", lang))

        flags["injection_flagged"] = looks_like_injection(pre.text)

        try:
            with trace.span("models"):
                probs, intent = await asyncio.gather(
                    self._infer(self.route_model.predict, pre.text),
                    self._infer(self.intent_model.predict, pre.text) if self.intent_model else self._none(),
                )
        except Exception:
            logger.exception("classifier inference failed")
            return make(status="error", action="clarify", error_code="model_error", retryable=False)

        decision = decide(probs, self.route_model.labels, self.config.tau_route)
        base.update(route=decision.route, route_confidence=decision.confidence,
                    intent=intent[0] if intent else None,
                    intent_confidence=intent[1] if intent else None)

        if decision.action == "refuse":
            reason = decision.refusal_reason or "policy"
            return make(status="refused", action="refuse", refusal_reason=reason,
                        answer=text_for(reason, lang))
        if decision.action == "clarify":
            return make(status="clarify", action="clarify", answer=text_for("clarify", lang))

        tool_fn = self.rag_tool if decision.action == "rag" else self.sql_tool
        outcome, err = await self._call_tool(decision.action, tool_fn, trace, pre.text, lang)
        if err is not None:
            code, retryable = err
            return make(status="error", action=decision.action, tool=decision.action,
                        error_code=code, retryable=retryable)
        if decision.action == "rag":
            return self._map_rag(outcome, make, lang)
        return self._map_sql(outcome, make, lang)

    async def _call_tool(self, name: str, fn: Tool, trace: Trace, query: str, lang: str):
        try:
            with trace.span(name):
                return await asyncio.wait_for(fn(query, lang), self.config.tool_timeout_s), None
        except asyncio.TimeoutError:
            logger.warning("%s tool timed out after %.1fs", name, self.config.tool_timeout_s)
            return None, ("timeout", True)
        except LLMUnavailable:
            logger.warning("%s tool: LLM unavailable", name)
            return None, ("llm_unavailable", True)
        except LLMError:
            logger.exception("%s tool: LLM error", name)
            return None, ("llm_error", False)
        except Exception:
            logger.exception("%s tool failed", name)
            return None, ("internal_error", False)

    def _map_rag(self, o: Any, make: Callable[..., AgentResult], lang: str) -> AgentResult:
        detail = {"gate_prob": o.gate_prob, "context_ids": o.context_ids,
                  "dropped_claims": o.dropped_claims, "degraded_reason": o.degraded_reason, "llm": o.llm}
        if o.status == "answered":
            if leaks_canary(o.answer):
                return make(status="refused", action="rag", tool="rag", refusal_reason="policy",
                            answer=text_for("policy", lang), detail=detail)
            return make(status="answered", action="rag", tool="rag", answer=o.answer,
                        citations=list(o.citations), degraded=bool(o.degraded), detail=detail)
        reason = o.refusal_reason or "no_evidence"
        return make(status="refused", action="rag", tool="rag", refusal_reason=reason,
                    answer=o.answer or text_for(reason, lang), detail=detail)

    def _map_sql(self, o: Any, make: Callable[..., AgentResult], lang: str) -> AgentResult:
        detail = {"llm_calls": o.llm_calls, "failure_codes": o.failure_codes, "sql_timings_ms": o.timings_ms}
        sql_fields = {"sql": o.sql, "columns": list(o.columns or []), "rows": list(o.rows or []),
                      "row_count": o.row_count, "truncated": bool(o.truncated)}
        base = {"action": "sql", "tool": "sql", "detail": detail}
        if o.status == "clarify":
            return make(status="clarify", answer=o.clarification or text_for("clarify", lang), **base)
        if o.status == "refused":
            reason = o.refusal_reason or "no_evidence"
            return make(status="refused", refusal_reason=reason,
                        answer=o.answer or text_for(reason, lang), **base)
        if leaks_canary(o.answer):
            return make(status="refused", refusal_reason="policy", answer=text_for("policy", lang), **base)
        if o.row_count == 0 and self.config.zero_rows_refuse:
            return make(status="refused", refusal_reason="no_evidence",
                        answer=text_for("no_rows", lang), **base, **sql_fields)
        return make(status="answered", answer=o.answer, **base, **sql_fields)