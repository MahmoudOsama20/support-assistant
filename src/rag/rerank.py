"""Cross-encoder reranker (bge-reranker-v2-m3). Scores are raw logits; use sigmoid() for a probability."""
from __future__ import annotations

import math

from rag.bm25 import Hit
from rag.dense import resolve_device

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CrossEncoderReranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto", max_length: int = 512,
                 batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
                      .to(self.device).eval())

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                scores.extend(self.model(**enc).logits.view(-1).float().cpu().tolist())
        return scores

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.score(query, [h.chunk.index_text for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))[:top_n]
        return [Hit(hits[i].chunk, scores[i]) for i in order]