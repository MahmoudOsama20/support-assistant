import json

import pytest
from pydantic import ValidationError

from agent.core import AgentResult
from agent.schemas import (RESPONSE_ADAPTER, ChatRequest, Clarification, ErrorResponse, RagAnswer, Refusal,
                           SqlAnswer, to_response)

CITE = {"doc_id": "kb_003", "chunk_id": "kb_003#c1", "title": "Transfer limits", "heading": "Limits",
        "status": "active", "snippet": "s", "extra_key": 1}


def result(**kw) -> AgentResult:
    d = dict(status="answered", action="rag", language="en", route="kb_question", route_confidence=0.96,
             answer="The limit is 5,000 EGP.", tool="rag", citations=[CITE],
             flags={"injection_flagged": False}, timings_ms={"total": 12.0})
    d.update(kw)
    return AgentResult(**d)


def test_rag_answer() -> None:
    r = to_response(result(), "rid1")
    assert isinstance(r, RagAnswer) and (r.type, r.status, r.tool) == ("rag_answer", "success", "rag")
    assert r.citations[0].chunk_id == "kb_003#c1" and r.request_id == "rid1"
    assert r.metadata.timings_ms == {"total": 12.0}


def test_sql_answer() -> None:
    r = to_response(result(action="sql", tool="sql", route="data_lookup", citations=[], sql="SELECT 1",
                           columns=["balance"], rows=[[84250.5]], row_count=1, answer="balance: 84250.5"), "rid")
    assert isinstance(r, SqlAnswer) and r.rows == [[84250.5]] and r.row_count == 1 and not r.truncated


def test_invalid_combinations_become_invalid_response() -> None:
    for bad in (result(citations=[]), result(tool=None)):
        out = to_response(bad, "rid")
        assert isinstance(out, ErrorResponse) and out.code == "invalid_response" and not out.retryable


def test_refusal_and_unknown_reason() -> None:
    kw = dict(status="refused", action="refuse", tool=None, citations=[], answer="Sorry.")
    ok = to_response(result(refusal_reason="out_of_scope", **kw), "r")
    assert isinstance(ok, Refusal) and ok.reason == "out_of_scope" and ok.message == "Sorry."
    bad = to_response(result(refusal_reason="whatever", **kw), "r")
    assert isinstance(bad, ErrorResponse) and bad.code == "invalid_response"


def test_sql_no_rows_refusal_keeps_sql() -> None:
    r = to_response(result(status="refused", action="sql", tool="sql", citations=[], answer="No rows.",
                           refusal_reason="no_evidence", sql="SELECT 1", row_count=0), "r")
    assert isinstance(r, Refusal) and (r.sql, r.row_count, r.tool) == ("SELECT 1", 0, "sql")


def test_clarify() -> None:
    r = to_response(result(status="clarify", action="clarify", tool=None, citations=[],
                           answer="Could you clarify?"), "r")
    assert isinstance(r, Clarification) and r.status == "clarify"


def test_error_mapping() -> None:
    r = to_response(result(status="error", action="rag", error_code="timeout", retryable=True,
                           answer="", citations=[]), "r")
    assert isinstance(r, ErrorResponse) and (r.code, r.retryable) == ("timeout", True) and r.message


def test_unknown_language_falls_back_to_en() -> None:
    assert to_response(result(language="other"), "r").language == "en"


def test_json_roundtrip_and_extra_forbidden() -> None:
    data = json.loads(to_response(result(), "rid").model_dump_json())
    assert data["type"] == "rag_answer"
    assert isinstance(RESPONSE_ADAPTER.validate_python(data), RagAnswer)
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        RESPONSE_ADAPTER.validate_python(data)


@pytest.mark.parametrize("q", ["", "x" * 2001])
def test_chat_request_limits(q: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query=q)


def test_chat_request_ok() -> None:
    assert ChatRequest(query="مرحبا").query == "مرحبا"