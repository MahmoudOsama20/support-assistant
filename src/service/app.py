"""Async FastAPI service around the support agent."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agent.schemas import (
    RESPONSE_ADAPTER,
    ChatRequest,
    ErrorResponse,
    new_request_id,
    to_response,
)
from rag.normalize import detect_language
from service.trace_log import TraceWriter, build_record

log = logging.getLogger("support.service")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = ROOT / "logs" / "traces.jsonl"
DEFAULT_DB_PATH = ROOT / "data" / "db" / "support.db"
DEFAULT_WARMUP_QUERY = "what's the weather like in Cairo"  # refused by the route model: no LLM call

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

ERROR_HTTP_STATUS = {
    "timeout": 504,
    "llm_unavailable": 503,
    "llm_error": 502,
    "invalid_response": 502,
    "model_error": 500,
    "internal_error": 500,
}


def error_response(request_id: str, query: str, code: str, message: str, retryable: bool) -> ErrorResponse:
    """Service-level error (timeout / crash) in the same schema as agent errors."""
    return ErrorResponse.model_validate(
        {
            "request_id": request_id,
            "language": detect_language(query),
            "intent": None,
            "intent_confidence": None,
            "route": "none",
            "route_confidence": 0.0,
            "metadata": {"flags": {}, "timings_ms": {}, "degraded": True},
            "type": "error",
            "status": "error",
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    )


def _build_real_agent() -> Any:
    # Heavy imports stay here so importing this module never loads torch.
    from agent.build import build_agent, pick_device
    from agent.core import AgentConfig

    db_path = Path(os.getenv("DB_PATH") or DEFAULT_DB_PATH)
    return build_agent(
        config=AgentConfig(),
        db_path=db_path,
        device=pick_device(),
        with_intent=True,
    )


async def _warm_up(agent: Any) -> None:
    query = os.getenv("WARMUP_QUERY", DEFAULT_WARMUP_QUERY)
    try:
        t0 = time.perf_counter()
        await agent.handle(query)
        log.info("warm-up done in %.0f ms", (time.perf_counter() - t0) * 1000)
    except Exception:
        log.exception("warm-up failed (continuing)")


def create_app(
    *,
    agent: Any | None = None,
    trace_path: Path | str | None = None,
    request_timeout_s: float | None = None,
) -> FastAPI:
    timeout_s = request_timeout_s if request_timeout_s is not None else float(os.getenv("REQUEST_TIMEOUT_S", "60"))
    trace = TraceWriter(trace_path or os.getenv("TRACE_PATH") or DEFAULT_TRACE_PATH)
    log_query = os.getenv("TRACE_LOG_QUERY", "0") == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle = None
        if agent is not None:
            app.state.agent = agent
        else:
            bundle = _build_real_agent()  # blocking, once, at startup
            app.state.agent = bundle.agent
            await _warm_up(app.state.agent)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            if bundle is not None:
                await bundle.aclose()

    app = FastAPI(title="Nile Wallet Support Assistant", lifespan=lifespan)
    app.state.ready = False

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if REQUEST_ID_RE.fullmatch(incoming) else new_request_id()
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok" if app.state.ready else "starting", "model_ready": bool(app.state.ready)}

    @app.post("/chat")
    async def chat(body: ChatRequest):
        rid = request_id_var.get()
        t0 = time.perf_counter()
        result = None
        try:
            result = await asyncio.wait_for(app.state.agent.handle(body.query), timeout=timeout_s)
            response = to_response(result, rid)
        except TimeoutError:
            log.warning("request %s hit the global timeout (%.1f s)", rid, timeout_s)
            response = error_response(rid, body.query, "timeout", "The request took too long. Please try again.", True)
        except Exception:
            log.exception("request %s crashed", rid)
            response = error_response(rid, body.query, "internal_error", "Internal error.", False)

        payload = RESPONSE_ADAPTER.dump_python(response, mode="json")
        http_status = 200 if payload["status"] != "error" else ERROR_HTTP_STATUS.get(payload.get("code"), 500)
        wall_ms = (time.perf_counter() - t0) * 1000

        try:
            await trace.write(
                build_record(
                    request_id=rid,
                    query=body.query,
                    wall_ms=wall_ms,
                    http_status=http_status,
                    payload=payload,
                    result=result,
                    log_query=log_query,
                )
            )
        except Exception:
            log.exception("trace write failed for %s", rid)

        log.info(
            "request %s route=%s status=%s http=%d total=%.0f ms",
            rid, payload.get("route"), payload["status"], http_status, wall_ms,
        )
        return JSONResponse(payload, status_code=http_status)

    return app


app = create_app()