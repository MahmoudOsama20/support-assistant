"""Torch-free metric helpers for intent classification."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def group_metrics(
    groups: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Metrics per group (e.g. per locale), preserving first-seen group order."""
    out: dict[str, dict[str, float]] = {}
    for group in dict.fromkeys(groups):
        idx = [i for i, g in enumerate(groups) if g == group]
        out[group] = {
            "n": len(idx),
            **compute_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }
    return out


def build_confusion(y_true: Sequence[int], y_pred: Sequence[int], num_labels: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))


def top_confused_pairs(cm: np.ndarray, labels: Sequence[str], k: int = 10) -> list[dict]:
    """Largest off-diagonal confusion cells, with share of the true class."""
    off_diagonal = cm - np.diag(np.diag(cm))
    rows, cols = np.nonzero(off_diagonal)
    pairs = sorted(
        ((int(cm[r, c]), int(r), int(c)) for r, c in zip(rows, cols)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    row_totals = cm.sum(axis=1)
    return [
        {
            "true": labels[r],
            "pred": labels[c],
            "count": n,
            "share_of_true": round(n / int(row_totals[r]), 4),
        }
        for n, r, c in pairs[:k]
    ]


def within_scenario_error_share(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    intent_to_scenario: Mapping[str, str],
) -> float:
    """Fraction of errors where the predicted intent is in the true intent's scenario."""
    errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    if not errors:
        return 0.0
    same = sum(
        1 for t, p in errors if intent_to_scenario[labels[t]] == intent_to_scenario[labels[p]]
    )
    return same / len(errors)


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Percentile bootstrap CIs for accuracy and macro-F1."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        accs.append(float(np.mean(yt[idx] == yp[idx])))
        f1s.append(float(f1_score(yt[idx], yp[idx], average="macro", zero_division=0)))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
    }