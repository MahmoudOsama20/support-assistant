import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from service.app import REQUEST_ID_RE, create_app
from service.trace_stats import percentile, summarize


def make_result(**overrides):
    # SimpleNamespace stands in for AgentResult: same attribute names, no model loading.
    fields = dict(
        status="clarify", action="clarify", language="en", route="none", route_confidence=0.0,
        answer="Could you clarify your question?", tool=None, refusal_reason=None,
        intent=None, intent_confidence=None, citations=[], sql=None, columns=[], rows=[],
        row_count=None, truncated=False, degraded=False, error_code=None, retryable=False,
        flags={"too_short": True}, timings_ms={"total": 1.0}, detail={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def rag_result(**overrides):
    base = dict(
        status="answered", action="rag", route="kb_question", route_confidence=0.99,
        answer="The limit is 5,000 EGP.", tool="rag", intent="qa_currency", intent_confidence=0.5,
        citations=[{"doc_id": "kb_003", "chunk_id": "kb_003#c1", "title": "Transfer limits",
                    "heading": "Daily limits", "status": "active", "snippet": "Limits by tier."}],
        flags={"too_short": False},
        detail={"gate_prob": 0.8, "context_ids": ["kb_003#c1"],
                "llm": {"cached": True, "latency_ms": 3.0, "prompt_tokens": 700, "completion_tokens": 90}},
    )
    base.update(overrides)
    return make_result(**base)


class FakeAgent:
    def __init__(self, result=None, delay=0.0, exc=None):
        self.result, self.delay, self.exc = result, delay, exc

    async def handle(self, raw_query):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.result


def read_trace(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_health(tmp_path):
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_ready": True}


def test_clarify_response_and_request_id_header(tmp_path):
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "help"})
    body = r.json()
    assert r.status_code == 200
    assert body["type"] == "clarification" and body["status"] == "clarify"
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_rag_answer_writes_trace_line(tmp_path):
    trace = tmp_path / "t.jsonl"
    app = create_app(agent=FakeAgent(rag_result()), trace_path=trace)
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "what is the daily transfer limit?"})
    assert r.status_code == 200 and r.json()["type"] == "rag_answer"
    (rec,) = read_trace(trace)
    assert rec["request_id"] == r.json()["request_id"]
    assert rec["route"] == "kb_question" and rec["status"] == "success"
    assert rec["citations"] == [{"doc_id": "kb_003", "chunk_id": "kb_003#c1"}]
    assert rec["llm_cached"] is True
    assert rec["detail"]["context_ids"] == ["kb_003#c1"]
    assert rec["total_latency_ms"] >= 0
    assert "query" not in rec  # query text is off by default


def test_client_request_id_is_honoured(tmp_path):
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "help"}, headers={"X-Request-ID": "abcdef123456"})
    assert r.headers["X-Request-ID"] == "abcdef123456"
    assert r.json()["request_id"] == "abcdef123456"


def test_malformed_request_id_is_replaced(tmp_path):
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "help"}, headers={"X-Request-ID": "bad id!"})
    assert r.headers["X-Request-ID"] != "bad id!"
    assert REQUEST_ID_RE.fullmatch(r.headers["X-Request-ID"])


def test_global_timeout_returns_error(tmp_path):
    trace = tmp_path / "t.jsonl"
    app = create_app(agent=FakeAgent(make_result(), delay=1.0), trace_path=trace, request_timeout_s=0.05)
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "what is the daily transfer limit?"})
    body = r.json()
    assert r.status_code == 504
    assert body["type"] == "error" and body["code"] == "timeout" and body["retryable"] is True
    (rec,) = read_trace(trace)
    assert rec["status"] == "error" and rec["error_code"] == "timeout" and rec["http_status"] == 504


def test_agent_crash_hides_traceback(tmp_path):
    app = create_app(agent=FakeAgent(exc=RuntimeError("secret path E:\\private")), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "what is the daily transfer limit?"})
    assert r.status_code == 500
    assert r.json()["code"] == "internal_error"
    assert "secret" not in r.text


@pytest.mark.parametrize("query", ["", "x" * 2001])
def test_query_validation(tmp_path, query):
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": query})
    assert r.status_code == 422


def test_invalid_response_maps_to_502(tmp_path):
    # RAG answer without citations must never reach the client as an answer.
    app = create_app(agent=FakeAgent(rag_result(citations=[])), trace_path=tmp_path / "t.jsonl")
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "what is the daily transfer limit?"})
    assert r.status_code == 502
    assert r.json()["code"] == "invalid_response"


def test_trace_failure_does_not_break_request(tmp_path):
    # trace_path is a directory -> opening it for append fails; the request must still succeed.
    app = create_app(agent=FakeAgent(make_result()), trace_path=tmp_path)
    with TestClient(app) as client:
        r = client.post("/chat", json={"query": "help"})
    assert r.status_code == 200


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10
    assert percentile(list(range(1, 101)), 0.95) == pytest.approx(95.05)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_summarize_groups_and_filters():
    recs = [
        {"route": "kb_question", "total_latency_ms": 100.0, "status": "success", "llm_cached": False},
        {"route": "kb_question", "total_latency_ms": 300.0, "status": "success", "llm_cached": True},
        {"route": "out_of_scope", "total_latency_ms": 20.0, "status": "refused", "llm_cached": None},
    ]
    s = summarize(recs)
    assert s["groups"]["ALL"]["n"] == 3
    assert s["groups"]["kb_question"]["p50_ms"] == 200.0
    assert s["llm_requests"] == 2 and s["llm_cached"] == 1
    s = summarize(recs, exclude_cached=True)
    assert s["groups"]["ALL"]["n"] == 2 and s["groups"]["kb_question"]["n"] == 1
    assert summarize(recs, skip_first=1)["groups"]["ALL"]["n"] == 2