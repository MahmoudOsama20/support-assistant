import json
from pathlib import Path

import numpy as np
import pytest

from route.train_route import (
    ROUTE_LABELS, ece, fit_temperature, load_route_rows, nll, softmax_np,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(i, label="kb_question", split="train", lang="en"):
    return {"id": f"{split}-{i}", "text": f"q{i}", "label": label, "language": lang, "split": split,
            "source": "synthetic", "ref": "f"}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_rows_ok(tmp_path):
    p = tmp_path / "route_train.jsonl"
    _write(p, [_row(0), _row(1, "unsafe_request", lang="ar")])
    rows = load_route_rows(p, "train")
    assert rows[1] == {"id": "train-1", "utt": "q1", "intent": "unsafe_request", "locale": "ar"}


def test_load_rows_rejects_unknown_label(tmp_path):
    p = tmp_path / "route_train.jsonl"
    _write(p, [_row(0, "weather")])
    with pytest.raises(ValueError):
        load_route_rows(p, "train")


def test_load_rows_rejects_wrong_split(tmp_path):
    p = tmp_path / "route_dev.jsonl"
    _write(p, [_row(0, split="train")])
    with pytest.raises(ValueError):
        load_route_rows(p, "dev")


def test_temperature_cools_overconfident_logits():
    rng = np.random.default_rng(0)
    n, k = 600, len(ROUTE_LABELS)
    y = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))  # ~72% accurate
    logits = np.zeros((n, k))
    logits[np.arange(n), pred] = 8.0  # very confident regardless of correctness
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert nll(logits, y, t) < nll(logits, y, 1.0)
    assert ece(softmax_np(logits, t), y) < ece(softmax_np(logits), y)


def test_ece_near_zero_when_confident_and_correct():
    y = np.arange(40) % 4
    logits = np.full((40, 4), -10.0)
    logits[np.arange(40), y] = 10.0
    assert ece(softmax_np(logits), y) < 1e-3


@pytest.mark.skipif(not (ROOT / "data/route/route_dev.jsonl").exists(), reason="route data not generated")
def test_real_dev_rows_load():
    rows = load_route_rows(ROOT / "data/route/route_dev.jsonl", "dev")
    assert len(rows) == 320
    assert {r["intent"] for r in rows} == set(ROUTE_LABELS)