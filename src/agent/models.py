"""Thin wrappers around the two fine-tuned classifiers (route model + Part A intent model).

Sanity check with real models (no RAG/SQL):
    python src/agent/models.py "what is the daily transfer limit?" "show tickets for Mona Adel" "what's the weather"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.policy import DEFAULT_TAU_ROUTE, decide  # noqa: E402
from agent.preprocess import preprocess  # noqa: E402
from classifier.model import load_classifier, predict_probs, resolve_device  # noqa: E402
from route.train_route import ROUTE_LABELS, collect_logits, softmax_np  # noqa: E402

ROUTE_DIR = PROJECT_ROOT / "results" / "runs" / "route-e5small" / "model"
INTENT_DIR = PROJECT_ROOT / "results" / "runs" / "e5-small-e15" / "model"


class RouteClassifier:
    """4-way route model; probabilities use the temperature fit on route dev."""

    def __init__(self, model_dir: Path, device: torch.device, max_length: int = 64) -> None:
        self.device, self.max_length = device, max_length
        self.model, self.tokenizer, labels = load_classifier(model_dir, device)
        self.labels: list[str] = list(labels)
        if self.labels != ROUTE_LABELS:
            raise ValueError(f"route labels {self.labels} != {ROUTE_LABELS}")
        self.temperature: float = json.loads(
            (Path(model_dir) / "temperature.json").read_text(encoding="utf-8"))["temperature"]

    def predict(self, text: str) -> list[float]:
        logits = collect_logits(self.model, self.tokenizer, [text], self.device, self.max_length, 1)
        return softmax_np(logits, self.temperature)[0].tolist()


class IntentClassifier:
    """Part A MASSIVE 60-way intent model (logged only; not used for routing)."""

    def __init__(self, model_dir: Path, device: torch.device, max_length: int = 64) -> None:
        self.device, self.max_length = device, max_length
        self.model, self.tokenizer, labels = load_classifier(model_dir, device)
        self.labels: list[str] = list(labels)

    def predict(self, text: str) -> tuple[str, float]:
        probs = predict_probs(self.model, self.tokenizer, [text], self.device, self.max_length, 1)[0]
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU_ROUTE)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    route = RouteClassifier(ROUTE_DIR, device)
    intent = IntentClassifier(INTENT_DIR, device)
    print(f"device={device} tau={args.tau} T={route.temperature:.3f} (first call includes warm-up)")
    for q in args.queries:
        pre = preprocess(q)
        t = time.perf_counter()
        probs = route.predict(pre.text)
        t_route = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        label, conf = intent.predict(pre.text)
        t_intent = (time.perf_counter() - t) * 1000
        d = decide(probs, route.labels, args.tau)
        dist = " ".join(f"{l}={p:.3f}" for l, p in zip(route.labels, probs))
        print(f"\n[{pre.language}] {pre.text}\n  route: {dist}\n  action={d.action} route={d.route} "
              f"conf={d.confidence:.3f} reason={d.refusal_reason}\n  intent={label} ({conf:.3f}) "
              f"| route {t_route:.1f} ms, intent {t_intent:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())