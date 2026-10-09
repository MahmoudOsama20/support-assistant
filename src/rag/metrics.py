"""Retrieval metrics and the calibration/eval case loader."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "data" / "kb" / "calibration_queries.jsonl"
REQUIRED = ("id", "language", "category", "query", "gold_doc_ids")


@dataclass(frozen=True)
class Case:
    id: str
    language: str
    category: str
    query: str
    gold_doc_ids: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.gold_doc_ids)


def load_cases(path: Path, known_doc_ids: set[str]) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"{path}:{n}: missing fields {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {row['id']}")
        if row["language"] not in ("en", "ar"):
            raise ValueError(f"{path}:{n}: language must be en or ar, got {row['language']!r}")
        unknown = sorted(set(row["gold_doc_ids"]) - known_doc_ids)
        if unknown:
            raise ValueError(f"{path}:{n}: unknown gold doc ids {unknown}")
        seen.add(row["id"])
        cases.append(Case(row["id"], row["language"], row["category"], row["query"],
                          tuple(row["gold_doc_ids"])))
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def first_gold_rank(ranked_doc_ids: Sequence[str], gold: set[str]) -> int | None:
    """1-based position of the first chunk whose doc is gold; None if absent."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold:
            return rank
    return None


def summarize(ranks: Sequence[int | None], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    if not ranks:
        raise ValueError("no ranks to summarize")
    n = len(ranks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    return out


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(random positive scores higher than random negative); ties count 0.5."""
    if not positive or not negative:
        raise ValueError("auc needs at least one positive and one negative score")
    wins = sum((p > q) + 0.5 * (p == q) for p in positive for q in negative)
    return wins / (len(positive) * len(negative))