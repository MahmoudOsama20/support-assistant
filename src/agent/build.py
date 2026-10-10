"""Wire the real models and tools into a SupportAgent (used by the CLI now, the FastAPI lifespan later)."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.core import AgentConfig, SupportAgent, Tool  # noqa: E402
from llm.client import DEFAULT_CACHE_DIR, LLMClient, LLMSettings  # noqa: E402
from llm.env import load_dotenv_file  # noqa: E402
from rag.answer import answer_question  # noqa: E402
from rag.context import CONTEXT_FETCH, DEFAULT_TAU, build_context  # noqa: E402
from sqltool.execute import fetch_sample_rows  # noqa: E402
from sqltool.generate import answer_sql_question  # noqa: E402
from sqltool.llm_adapter import make_complete_from  # noqa: E402
from sqltool.schema import build_schema_prompt  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "data" / "db" / "support.db"


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_rag_tool(retriever, llm, *, tau: float = DEFAULT_TAU) -> Tool:
    """retrieve (blocking, one at a time) -> evidence gate -> grounded answer."""
    gate = asyncio.Semaphore(1)  # shared encoder/reranker thread-safety is UNVERIFIED: serialize retrieval

    async def rag_tool(query: str, language: str):
        async with gate:
            res = await asyncio.to_thread(retriever.retrieve, query, CONTEXT_FETCH)
        ctx = build_context(res.final, query, tau=tau)
        return await answer_question(query, ctx, llm)

    return rag_tool


def build_sql_tool(db_path: Path, complete, schema_prompt: str, *, timeout_s: float = 2.0) -> Tool:
    async def sql_tool(query: str, language: str):
        lang = "ar" if language == "ar" else "en"
        return await answer_sql_question(query, schema_prompt, db_path, complete,
                                         language=lang, timeout_s=timeout_s)

    return sql_tool


@dataclass
class AgentBundle:
    agent: SupportAgent
    llm: LLMClient

    async def aclose(self) -> None:
        await self.llm.aclose()


def build_agent(*, config: AgentConfig = AgentConfig(), db_path: Path = DEFAULT_DB,
                device: str | None = None, with_intent: bool = True) -> AgentBundle:
    load_dotenv_file()
    settings = LLMSettings.from_env()
    if not settings.api_key:
        raise RuntimeError("no API key: set LLM_API_KEY or GROQ_API_KEY (in .env or the environment)")
    if not db_path.is_file():
        raise FileNotFoundError(f"{db_path} missing: run python src/supportdb/seed.py")
    dev = device or pick_device()

    # heavy imports live here so importing this module stays cheap
    from agent.models import INTENT_DIR, ROUTE_DIR, IntentClassifier, RouteClassifier
    from rag.bm25 import BM25Index
    from rag.chunking import load_chunks
    from rag.dense import DEFAULT_CACHE, BgeM3Encoder, DenseIndex
    from rag.rerank import CrossEncoderReranker
    from rag.retriever import HybridRetriever

    chunks = load_chunks()
    retriever = HybridRetriever(
        BM25Index(chunks),
        DenseIndex(chunks, BgeM3Encoder(device=dev), cache_path=DEFAULT_CACHE),
        CrossEncoderReranker(device=dev),
        fetch_k=20, rerank_top=20,
    )
    route = RouteClassifier(ROUTE_DIR, dev)
    intent = IntentClassifier(INTENT_DIR, dev) if with_intent else None
    schema_prompt = build_schema_prompt(fetch_sample_rows(db_path))

    llm = LLMClient(settings, cache_dir=DEFAULT_CACHE_DIR)  # last, so a failed load leaks nothing
    agent = SupportAgent(
        route, intent,
        build_rag_tool(retriever, llm),
        build_sql_tool(db_path, make_complete_from(llm), schema_prompt),
        config,
    )
    return AgentBundle(agent=agent, llm=llm)