"""Deterministic route policy: route-model probabilities -> action.

Pure functions, no model or I/O. The LLM never chooses the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

LABEL_TO_ACTION = {
    "kb_question": "rag",
    "data_lookup": "sql",
    "out_of_scope": "refuse",
    "unsafe_request": "refuse",
}
REFUSAL_REASON = {"out_of_scope": "out_of_scope", "unsafe_request": "unsafe_request"}
TOOL_ACTIONS = ("rag", "sql")
DEFAULT_TAU_ROUTE = 0.95  # frozen from route DEV sweep (amended rule); never retune on heldout or eval cases


@dataclass(frozen=True)
class RouteDecision:
    action: str                 # rag | sql | clarify | refuse
    route: str                  # top-1 route label (even when clarifying)
    confidence: float           # top-1 probability
    refusal_reason: str | None  # set only when action == "refuse"
    basis: str                  # "model" | "low_confidence"


def decide(probs: Sequence[float], labels: Sequence[str], tau_route: float) -> RouteDecision:
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} probabilities for {len(labels)} labels")
    unknown = set(labels) - set(LABEL_TO_ACTION)
    if unknown:
        raise ValueError(f"unknown route labels: {sorted(unknown)}")
    best = max(range(len(probs)), key=lambda i: probs[i])
    label, conf = labels[best], float(probs[best])
    if conf < tau_route:
        return RouteDecision("clarify", label, conf, None, "low_confidence")
    action = LABEL_TO_ACTION[label]
    reason = REFUSAL_REASON.get(label) if action == "refuse" else None
    return RouteDecision(action, label, conf, reason, "model")


def threshold_sweep(
    probs: Any, true_labels: Sequence[str], labels: Sequence[str], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Run the real decide() at each threshold and count the outcomes that matter."""
    rows = []
    n = len(true_labels)
    tool_bound = [LABEL_TO_ACTION[t] in TOOL_ACTIONS for t in true_labels]
    n_tool = sum(tool_bound)
    for tau in thresholds:
        decisions = [decide(p, labels, tau) for p in probs]
        accepted = [(d, t) for d, t in zip(decisions, true_labels) if d.action != "clarify"]
        correct = sum(d.route == t for d, t in accepted)
        over_clarify = sum(d.action == "clarify" and tb for d, tb in zip(decisions, tool_bound))
        tool_misroute = sum(
            tb and d.action in TOOL_ACTIONS and d.action != LABEL_TO_ACTION[t]
            for d, t, tb in zip(decisions, true_labels, tool_bound)
        )
        over_refusal = sum(tb and d.action == "refuse" for d, tb in zip(decisions, tool_bound))
        unsafe_leak = sum(t == "unsafe_request" and d.action in TOOL_ACTIONS for d, t in zip(decisions, true_labels))
        oos_leak = sum(t == "out_of_scope" and d.action in TOOL_ACTIONS for d, t in zip(decisions, true_labels))
        rows.append({
            "tau": float(tau),
            "clarified": n - len(accepted),
            "accepted_accuracy": correct / len(accepted) if accepted else float("nan"),
            "over_clarify": over_clarify,
            "over_clarify_rate": over_clarify / n_tool if n_tool else 0.0,
            "tool_misroute": tool_misroute,
            "over_refusal": over_refusal,
            "unsafe_leak": unsafe_leak,
            "oos_leak": oos_leak,
        })
    return rows