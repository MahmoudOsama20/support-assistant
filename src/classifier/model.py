"""Model/tokenizer loading and batched prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.dataset import load_labels  # noqa: E402


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(choice)


def build_model(model_name: str, labels: Sequence[str]):
    """Pretrained encoder + a new classification head sized to our labels."""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )


def load_classifier(model_dir: Path, device: torch.device):
    """Load a fine-tuned checkpoint saved by train.py. Returns (model, tokenizer, labels)."""
    model_dir = Path(model_dir)
    if not (model_dir / "labels.json").is_file():
        raise FileNotFoundError(f"No trained model in {model_dir} (labels.json missing)")
    labels = load_labels(model_dir / "labels.json")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, labels


@torch.inference_mode()
def predict_probs(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int = 64,
    batch_size: int = 128,
) -> torch.Tensor:
    """Softmax probabilities, shape (len(texts), num_labels), on CPU, fp32."""
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        chunks.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu())
    return torch.cat(chunks)