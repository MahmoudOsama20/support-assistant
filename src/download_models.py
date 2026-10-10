"""Warm the Hugging Face cache (run once, with network):

docker compose run --rm -e HF_HUB_OFFLINE=0 api python src/download_models.py
"""
from __future__ import annotations

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ENCODER = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    # Same loading path as the service, so the same files (incl. converted safetensors) are cached.
    for name, model_cls in ((ENCODER, AutoModel), (RERANKER, AutoModelForSequenceClassification)):
        print(f"downloading {name} ...", flush=True)
        AutoTokenizer.from_pretrained(name)
        model_cls.from_pretrained(name)
        print(f"cached {name}", flush=True)


if __name__ == "__main__":
    main()