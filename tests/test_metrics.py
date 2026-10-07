from __future__ import annotations

import numpy as np
import pytest

from classifier.metrics import (
    bootstrap_ci,
    compute_metrics,
    group_metrics,
    top_confused_pairs,
    within_scenario_error_share,
)


def test_compute_metrics_perfect() -> None:
    m = compute_metrics([0, 1, 2], [0, 1, 2])
    assert m == {"accuracy": 1.0, "macro_f1": 1.0}


def test_compute_metrics_known_case() -> None:
    m = compute_metrics([0, 1, 1, 2], [0, 1, 2, 2])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(7 / 9)


def test_group_metrics_splits_by_group() -> None:
    out = group_metrics(["a", "a", "b", "b"], [0, 1, 0, 1], [0, 1, 1, 0])
    assert out["a"]["accuracy"] == 1.0 and out["b"]["accuracy"] == 0.0
    assert out["a"]["n"] == out["b"]["n"] == 2


def test_top_confused_pairs_orders_by_count() -> None:
    cm = np.array([[5, 2, 0], [1, 4, 0], [0, 3, 6]])
    pairs = top_confused_pairs(cm, ["a", "b", "c"], k=2)
    assert (pairs[0]["true"], pairs[0]["pred"], pairs[0]["count"]) == ("c", "b", 3)
    assert pairs[0]["share_of_true"] == pytest.approx(3 / 9, abs=1e-4)
    assert (pairs[1]["true"], pairs[1]["pred"]) == ("a", "b")


def test_within_scenario_error_share() -> None:
    labels = ["alarm_set", "alarm_query", "music_play"]
    scenario = {"alarm_set": "alarm", "alarm_query": "alarm", "music_play": "music"}
    share = within_scenario_error_share([0, 0, 2], [1, 2, 0], labels, scenario)
    assert share == pytest.approx(1 / 3)


def test_bootstrap_ci_perfect_predictions() -> None:
    y = [0, 1, 2] * 10
    ci = bootstrap_ci(y, y, n_boot=50, seed=1)
    assert ci["accuracy"] == [1.0, 1.0] and ci["macro_f1"] == [1.0, 1.0]


def test_bootstrap_ci_brackets_point_estimate() -> None:
    y_true = [0, 1] * 50
    y_pred = list(y_true)
    for i in range(0, 100, 10):
        y_pred[i] = 1 - y_pred[i]  # 10% errors
    lo, hi = bootstrap_ci(y_true, y_pred, n_boot=200, seed=1)["accuracy"]
    assert lo < 0.9 < hi