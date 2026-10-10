"""JSONL request tracing: one line per /chat request."""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _append(self, line: str) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        await asyncio.to_thread(self._append, line)


def build_record(
    *,
    request_id: str,
    query: str,
    wall_ms: float,
    http_status: int,
    payload: dict[str, Any],
    result: Any | None,
    log_query: bool = False,
) -> dict[str, Any]:
    """Flatten the public response + AgentResult internals into one trace record."""
    meta = payload.get("metadata") or {}
    detail = getattr(result, "detail", None) or {}
    llm = detail.get("llm") if isinstance(detail, dict) else None
    llm = llm if isinstance(llm, dict) else None
    citations = [
        {"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")}
        for c in (getattr(result, "citations", None) or [])
    ]
    record: dict[str, Any] = {
        "request_id": request_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "query_chars": len(query),
        "language": payload.get("language"),
        "intent": payload.get("intent"),
        "intent_confidence": payload.get("intent_confidence"),
        "route": payload.get("route"),
        "route_confidence": payload.get("route_confidence"),
        "type": payload.get("type"),
        "status": payload.get("status"),
        "tool": payload.get("tool"),
        "reason": payload.get("reason"),
        "error_code": payload.get("code"),
        "http_status": http_status,
        "flags": meta.get("flags"),
        "timings_ms": meta.get("timings_ms"),
        "total_latency_ms": round(wall_ms, 2),
        "degraded": meta.get("degraded"),
        "sql": payload.get("sql"),
        "row_count": payload.get("row_count"),
        "citations": citations,
        "llm_cached": llm.get("cached") if llm else None,
        "llm_latency_ms": llm.get("latency_ms") if llm else None,
        "detail": detail,
    }
    if log_query:
        record["query"] = query[:300]
    return record