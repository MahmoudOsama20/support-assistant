import asyncio
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import agent.build as build
from llm.env import load_dotenv_file
from sqltool.llm_adapter import make_complete_from


def test_sql_tool_passes_args_and_normalizes_language(monkeypatch) -> None:
    seen: dict = {}

    async def fake(question, schema_prompt, db_path, complete, *, language, timeout_s):
        seen.update(question=question, schema=schema_prompt, db=db_path, language=language, timeout=timeout_s)
        return "outcome"

    monkeypatch.setattr(build, "answer_sql_question", fake)
    tool = build.build_sql_tool(Path("x.db"), complete=None, schema_prompt="SCHEMA", timeout_s=1.5)
    assert asyncio.run(tool("balance of Mona Adel", "ar")) == "outcome"
    assert seen == {"question": "balance of Mona Adel", "schema": "SCHEMA", "db": Path("x.db"),
                    "language": "ar", "timeout": 1.5}
    asyncio.run(tool("q", "other"))
    assert seen["language"] == "en"


class SlowRetriever:
    def __init__(self) -> None:
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def retrieve(self, query: str, final_k: int):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return SimpleNamespace(final=[query])


def test_rag_tool_serializes_retrieval_and_uses_gate_tau(monkeypatch) -> None:
    taus: list[float] = []

    def fake_ctx(hits, query, tau):
        taus.append(tau)
        return SimpleNamespace(hits=hits)

    async def fake_answer(query, ctx, llm):
        return f"answer:{query}:{ctx.hits}"

    monkeypatch.setattr(build, "build_context", fake_ctx)
    monkeypatch.setattr(build, "answer_question", fake_answer)
    retriever = SlowRetriever()
    tool = build.build_rag_tool(retriever, llm=None, tau=0.07)

    async def go():
        return await asyncio.gather(*(tool(f"q{i}", "en") for i in range(4)))

    out = asyncio.run(go())
    assert out == [f"answer:q{i}:['q{i}']" for i in range(4)]
    assert retriever.peak == 1 and taus == [0.07] * 4


def test_make_complete_from_flattens_and_returns_text() -> None:
    class FakeClient:
        async def complete(self, system, user):
            self.args = (system, user)
            return SimpleNamespace(text="SELECT 1")

    client = FakeClient()
    complete = make_complete_from(client)
    out = asyncio.run(complete([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]))
    assert out == "SELECT 1" and client.args == ("S", "Q")


def test_load_dotenv_file_sets_missing_but_keeps_existing(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('A_TEST_KEY="abc"\nexport B_TEST_KEY=2\n# comment\n', encoding="utf-8")
    monkeypatch.delenv("A_TEST_KEY", raising=False)
    monkeypatch.setenv("B_TEST_KEY", "keep")
    try:
        load_dotenv_file(env)
        assert os.environ["A_TEST_KEY"] == "abc" and os.environ["B_TEST_KEY"] == "keep"
    finally:
        os.environ.pop("A_TEST_KEY", None)