"""Offline tests for the LLM baseline helpers (no network calls)."""

from __future__ import annotations

import numpy as np
import pytest

from classifier.dataset import DatasetError, build_labels, load_massive
from classifier.llm_baseline import (
    build_messages,
    cache_key,
    latency_stats,
    parse_label,
    parse_retry_after,
    sample_ids,
)
from classifier.metrics import bootstrap_ci, compute_metrics_present

LABELS = ["alarm_query", "alarm_set", "calendar_set", "play_music", "qa_factoid"]


# ---- parser ---------------------------------------------------------------- #
def test_parse_exact() -> None:
    assert parse_label("alarm_set", LABELS) == ("alarm_set", "exact")


@pytest.mark.parametrize("raw", [' "Alarm_Set". ', "`alarm_set`", "**alarm_set**", "ALARM_SET\n"])
def test_parse_normalized(raw: str) -> None:
    label, method = parse_label(raw, LABELS)
    assert label == "alarm_set" and method in ("exact", "normalized")


def test_parse_contained_in_sentence() -> None:
    assert parse_label("The intent is alarm_set.", LABELS) == ("alarm_set", "contained")


def test_parse_ambiguous_is_invalid() -> None:
    assert parse_label("alarm_set or alarm_query", LABELS) == (None, "invalid")


def test_parse_fuzzy_is_flagged() -> None:
    assert parse_label("alarm_sett", LABELS) == ("alarm_set", "fuzzy")


@pytest.mark.parametrize("raw", ["I don't know", "", "   ", None])
def test_parse_invalid(raw) -> None:
    assert parse_label(raw, LABELS) == (None, "invalid")


def test_parse_strips_think_blocks() -> None:
    assert parse_label("<think>maybe alarm_query</think>\nalarm_set", LABELS) == ("alarm_set", "exact")


# ---- prompt / cache key ------------------------------------------------------ #
def test_build_messages_lists_every_label_and_keeps_utterance() -> None:
    utt = "صحيني خمسة الفجر"
    messages = build_messages(utt, LABELS)
    assert messages[0]["role"] == "system" and messages[1] == {"role": "user", "content": utt}
    assert all(label in messages[0]["content"] for label in LABELS)


def test_cache_key_is_stable_and_sensitive() -> None:
    m = build_messages("wake me up", LABELS)
    base = cache_key("model-a", m, 0.0, 24)
    assert base == cache_key("model-a", m, 0.0, 24)
    assert base != cache_key("model-b", m, 0.0, 24)
    assert base != cache_key("model-a", build_messages("other", LABELS), 0.0, 24)
    assert base != cache_key("model-a", m, 0.7, 24)
    assert base != cache_key("model-a", m, 0.0, 32)


# ---- sampling ------------------------------------------------------------------ #
def _records() -> dict[str, list[dict[str, str]]]:
    def rec(i: str, loc: str, part: str) -> dict[str, str]:
        return {"id": i, "locale": loc, "partition": part, "scenario": "s", "intent": "s_i", "utt": "x"}
    data = {}
    for loc in ("ar-SA", "en-US"):
        data[loc] = [rec(str(i), loc, "test") for i in range(100)]
        data[loc] += [rec(f"t{i}", loc, "train") for i in range(20)]
    data["ar-SA"].append(rec("only_ar", "ar-SA", "test"))
    return data


def test_sample_ids_deterministic_nested_and_common_only() -> None:
    data = _records()
    ten = sample_ids(data, "test", 10, seed=42)
    assert ten == sample_ids(data, "test", 10, seed=42)
    assert sample_ids(data, "test", 5, seed=42) == ten[:5]
    assert len(set(ten)) == 10
    everything = sample_ids(data, "test", 1000, seed=42)
    assert "only_ar" not in everything and not any(i.startswith("t") for i in everything)
    assert len(everything) == 100
    assert ten != sample_ids(data, "test", 10, seed=7)


# ---- stats / metrics ----------------------------------------------------------- #
def test_latency_stats() -> None:
    stats = latency_stats([1, 2, 3, 4, 5])
    assert stats["p50"] == pytest.approx(3.0) and stats["p95"] == pytest.approx(4.8)
    assert latency_stats([]) is None


def test_parse_retry_after() -> None:
    assert parse_retry_after({"retry-after": "7"}) == 7.0
    assert parse_retry_after({}) is None
    assert parse_retry_after({"retry-after": "soon"}) is None


def test_metrics_present_has_no_phantom_class_for_invalid_outputs() -> None:
    m = compute_metrics_present([0, 0, 1, 1], [0, -1, 1, 1])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(5 / 6)


def test_bootstrap_present_only_perfect_predictions() -> None:
    y = [0, 1, 2] * 10
    ci = bootstrap_ci(y, y, n_boot=30, seed=1, present_only=True)
    assert ci["accuracy"] == [1.0, 1.0] and ci["macro_f1"] == [1.0, 1.0]


# ---- real data ------------------------------------------------------------------ #
def test_real_prompt_contains_all_60_labels() -> None:
    try:
        labels = build_labels(load_massive())
    except DatasetError as exc:
        pytest.fail(str(exc), pytrace=False)
    system = build_messages("hello", labels)[0]["content"]
    assert len(labels) == 60 and all(label in system for label in labels)