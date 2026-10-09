import numpy as np
import pytest

from agent.policy import decide, threshold_sweep
from agent.preprocess import MAX_QUERY_CHARS, preprocess

LABELS = ["kb_question", "data_lookup", "out_of_scope", "unsafe_request"]


# ---- preprocess ----
def test_nfkc_whitespace_and_hidden_chars():
    p = preprocess("  ＡＢＣ \u200b  how\n\tare you\u202e ")
    assert p.text == "ABC how are you"
    assert not p.truncated
    assert p.original.startswith("  ")


def test_language_detection():
    assert preprocess("كيف أفتح حساب جديد").language == "ar"
    assert preprocess("how do I open an account").language == "en"


def test_truncation():
    p = preprocess("a" * (MAX_QUERY_CHARS + 50))
    assert p.truncated and len(p.text) == MAX_QUERY_CHARS


def test_empty_after_cleaning():
    assert preprocess(" \u200b\n ").is_empty


def test_non_string_rejected():
    with pytest.raises(TypeError):
        preprocess(None)  # type: ignore[arg-type]


# ---- decide ----
@pytest.mark.parametrize("probs,action,reason", [
    ([0.9, 0.05, 0.03, 0.02], "rag", None),
    ([0.05, 0.9, 0.03, 0.02], "sql", None),
    ([0.05, 0.03, 0.9, 0.02], "refuse", "out_of_scope"),
    ([0.02, 0.03, 0.05, 0.9], "refuse", "unsafe_request"),
])
def test_decide_actions(probs, action, reason):
    d = decide(probs, LABELS, tau_route=0.5)
    assert (d.action, d.refusal_reason, d.basis) == (action, reason, "model")


def test_low_confidence_clarifies_and_keeps_top_route():
    d = decide([0.4, 0.3, 0.2, 0.1], LABELS, tau_route=0.5)
    assert d.action == "clarify" and d.route == "kb_question" and d.basis == "low_confidence"
    assert d.refusal_reason is None


def test_decide_validates_inputs():
    with pytest.raises(ValueError):
        decide([0.5, 0.5], LABELS, 0.5)
    with pytest.raises(ValueError):
        decide([0.5, 0.5], ["kb_question", "weather"], 0.5)


# ---- sweep ----
def test_threshold_sweep_counts():
    probs = np.array([
        [0.95, 0.03, 0.01, 0.01],  # kb, correct
        [0.40, 0.35, 0.15, 0.10],  # data_lookup, low confidence
        [0.10, 0.80, 0.05, 0.05],  # unsafe predicted as sql -> leak at low tau
        [0.05, 0.05, 0.85, 0.05],  # kb predicted out_of_scope -> over-refusal
    ])
    true = ["kb_question", "data_lookup", "unsafe_request", "kb_question"]
    low, high = threshold_sweep(probs, true, LABELS, [0.0, 0.9])
    assert (low["clarified"], low["unsafe_leak"], low["over_refusal"], low["over_clarify"]) == (0, 1, 1, 0)
    assert low["tool_misroute"] == 1  # row 2: data_lookup true, kb predicted
    assert (high["clarified"], high["unsafe_leak"], high["over_refusal"]) == (3, 0, 0)
    assert high["over_clarify"] == 2 and high["accepted_accuracy"] == 1.0