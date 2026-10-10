"""Public response schema (Pydantic v2) and the AgentResult -> response mapper."""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.core import AgentResult  # noqa: E402

logger = logging.getLogger("agent.schemas")

Language = Literal["ar", "en"]
Tool = Literal["rag", "sql"]
RefusalReason = Literal["no_evidence", "out_of_scope", "unsafe_request", "policy"]
ErrorCode = Literal["timeout", "llm_unavailable", "llm_error", "internal_error", "model_error", "invalid_response"]

ERROR_MESSAGES: dict[str, str] = {
    "timeout": "The request timed out. Please try again.",
    "llm_unavailable": "The language model is temporarily unavailable. Please try again shortly.",
    "llm_error": "The language model returned an error.",
    "internal_error": "Internal error.",
    "model_error": "A classifier failed to run.",
    "invalid_response": "The system produced an invalid response and withheld it.",
}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flags: dict[str, bool] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False


class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    doc_id: str
    chunk_id: str
    title: str = ""
    heading: str | None = None
    status: str | None = None
    snippet: str = ""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    language: Language
    intent: str | None = None            # Part A model output: logged only, never used for routing
    intent_confidence: float | None = None
    route: str
    route_confidence: float
    metadata: Metadata


class RagAnswer(_Base):
    type: Literal["rag_answer"] = "rag_answer"
    status: Literal["success"] = "success"
    tool: Literal["rag"] = "rag"
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class SqlAnswer(_Base):
    type: Literal["sql_answer"] = "sql_answer"
    status: Literal["success"] = "success"
    tool: Literal["sql"] = "sql"
    answer: str = Field(min_length=1)
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int = Field(ge=0)
    truncated: bool


class Clarification(_Base):
    type: Literal["clarification"] = "clarification"
    status: Literal["clarify"] = "clarify"
    message: str = Field(min_length=1)
    tool: Tool | None = None


class Refusal(_Base):
    type: Literal["refusal"] = "refusal"
    status: Literal["refused"] = "refused"
    reason: RefusalReason
    message: str = Field(min_length=1)
    tool: Tool | None = None
    sql: str | None = None
    row_count: int | None = None


class ErrorResponse(_Base):
    type: Literal["error"] = "error"
    status: Literal["error"] = "error"
    code: ErrorCode
    message: str
    retryable: bool


ChatResponse = Annotated[
    Union[RagAnswer, SqlAnswer, Clarification, Refusal, ErrorResponse],
    Field(discriminator="type"),
]
RESPONSE_ADAPTER: TypeAdapter = TypeAdapter(ChatResponse)


def new_request_id() -> str:
    return uuid.uuid4().hex


def _lang(r: AgentResult) -> str:
    return "ar" if r.language == "ar" else "en"


def _base(r: AgentResult, request_id: str) -> dict[str, Any]:
    return dict(
        request_id=request_id, language=_lang(r),
        intent=r.intent, intent_confidence=r.intent_confidence,
        route=r.route, route_confidence=r.route_confidence,
        metadata=Metadata(flags=r.flags, timings_ms=r.timings_ms, degraded=r.degraded),
    )


def _map(r: AgentResult, request_id: str) -> ChatResponse:
    base = _base(r, request_id)
    if r.status == "answered":
        if r.tool == "rag":
            return RagAnswer(answer=r.answer, citations=[Citation.model_validate(c) for c in r.citations], **base)
        if r.tool == "sql":
            return SqlAnswer(answer=r.answer, sql=r.sql, columns=r.columns, rows=r.rows,
                             row_count=r.row_count, truncated=r.truncated, **base)
        raise ValueError(f"answered result without a tool: {r.tool!r}")
    if r.status == "clarify":
        return Clarification(message=r.answer, tool=r.tool, **base)
    if r.status == "refused":
        return Refusal(reason=r.refusal_reason, message=r.answer, tool=r.tool,
                       sql=r.sql, row_count=r.row_count, **base)
    if r.status == "error":
        code = r.error_code or "internal_error"
        return ErrorResponse(code=code, message=ERROR_MESSAGES.get(code, "Request failed."),
                             retryable=r.retryable, **base)
    raise ValueError(f"unknown status {r.status!r}")


def to_response(r: AgentResult, request_id: str) -> ChatResponse:
    """Map and validate; an invalid combination becomes an `invalid_response` error, never a bad answer."""
    try:
        return _map(r, request_id)
    except ValueError:  # pydantic.ValidationError subclasses ValueError (pydantic-core)
        logger.exception("response validation failed (request_id=%s)", request_id)
        return ErrorResponse(
            request_id=request_id, language=_lang(r), route="none", route_confidence=0.0,
            metadata=Metadata(), code="invalid_response",
            message=ERROR_MESSAGES["invalid_response"], retryable=False,
        )