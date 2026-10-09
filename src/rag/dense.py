"""Dense retrieval: bge-m3 CLS embeddings, exact numpy search, on-disk cache with a staleness check.

torch/transformers are imported lazily so the pure parts (VectorIndex, DenseIndex with a fake
encoder) can be unit-tested without loading a model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rag.bm25 import Hit
from rag.chunking import Chunk

if TYPE_CHECKING:
    import torch

MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "index" / "dense_bge-m3.npz"


def resolve_device(name: str = "auto") -> torch.device:
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


class Encoder(Protocol):
    name: str
    max_length: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class BgeM3Encoder:
    """bge-m3 dense embedding = L2-normalised CLS vector (no instruction prefix needed)."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto", max_length: int = 512,
                 batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if not texts:
            raise ValueError("encode() needs at least one text")
        batches = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0].float()
                batches.append(F.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(batches).astype(np.float32)


class VectorIndex:
    """Exact cosine search over L2-normalised rows (dot product). Ties keep corpus order."""

    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"expected a non-empty (n, d) matrix, got shape {matrix.shape}")
        self.matrix = matrix.astype(np.float32, copy=False)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if query.shape != (self.matrix.shape[1],):
            raise ValueError(f"query shape {query.shape} does not match dimension {self.matrix.shape[1]}")
        scores = self.matrix @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]


def _fingerprint(name: str, max_length: int, chunks: list[Chunk]) -> str:
    h = hashlib.sha256(f"{name}|{max_length}".encode())
    for c in chunks:
        h.update(c.chunk_id.encode() + b"\0" + c.index_text.encode() + b"\0")
    return h.hexdigest()


def _load_cache(path: Path, fingerprint: str, n_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"]) != fingerprint or data["vectors"].shape[0] != n_rows:
                return None
            return data["vectors"]
    except (OSError, ValueError, KeyError):
        return None  # unreadable cache -> rebuild


class DenseIndex:
    def __init__(self, chunks: list[Chunk], encoder: Encoder, cache_path: Path | None = None) -> None:
        if not chunks:
            raise ValueError("cannot build a dense index from zero chunks")
        self.chunks = list(chunks)
        self.encoder = encoder
        fingerprint = _fingerprint(encoder.name, encoder.max_length, self.chunks)
        matrix = _load_cache(cache_path, fingerprint, len(self.chunks)) if cache_path else None
        if matrix is None:
            matrix = encoder.encode([c.index_text for c in self.chunks])
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, vectors=matrix, fingerprint=np.array(fingerprint))
        self._index = VectorIndex(matrix)

    def search(self, query: str, k: int = 20) -> list[Hit]:
        vector = self.encoder.encode([query])[0]
        return [Hit(self.chunks[i], score) for i, score in self._index.search(vector, k)]