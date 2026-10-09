"""Turn reranked hits into the LLM context: drop superseded chunks, apply the evidence gate."""
from __future__ import annotations

from dataclasses import dataclass

from rag.bm25 import Hit
from rag.chunking import Chunk
from rag.normalize import normalize_text
from rag.rerank import sigmoid

# tau: geometric midpoint of the gap between the lowest answerable (0.181) and the highest
# unanswerable-below-it (0.017) top-1 reranker probability on the 24 calibration queries
# (results/rag/retrieval_calibration.json). Frozen; do not retune on later runs.
DEFAULT_TAU = 0.05
CONTEXT_FETCH = 6  # reranked chunks requested from the retriever
CONTEXT_K = 4      # chunks passed to the LLM


@dataclass(frozen=True)
class ContextDecision:
    passed: bool
    gate_value: float | None  # None when no usable chunk remained
    tau: float
    chunks: list[Chunk]
    dropped_superseded: list[str]


def build_context(hits: list[Hit], query: str, *, tau: float = DEFAULT_TAU, gate_is_logit: bool = True,
                  k: int = CONTEXT_K) -> ContextDecision:
    """A superseded chunk is kept only if the query mentions its effective year (e.g. '2025')."""
    q = normalize_text(query)
    kept: list[Hit] = []
    dropped: list[str] = []
    for hit in hits:
        c = hit.chunk
        if c.status != "active" and c.effective_date[:4] not in q:
            dropped.append(c.chunk_id)
        else:
            kept.append(hit)
    if not kept:
        return ContextDecision(False, None, tau, [], dropped)
    value = sigmoid(kept[0].score) if gate_is_logit else kept[0].score
    return ContextDecision(value >= tau, value, tau, [h.chunk for h in kept[:k]], dropped)


def format_context(chunks: list[Chunk]) -> str:
    """Passages are delimited as untrusted data; superseded ones are labelled."""
    parts = [f'<passage id="{c.chunk_id}" status="{c.status}">\n{c.heading}\n{c.text}\n</passage>'
             for c in chunks]
    return "<context>\n" + "\n".join(parts) + "\n</context>"