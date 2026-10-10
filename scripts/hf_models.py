#!/usr/bin/env python3
"""Upload / fetch the two classifiers (weights are not in git).

python scripts/hf_models.py upload --repo-id USER/nile-wallet-classifiers [--public]   # needs $env:HF_TOKEN (write)
python scripts/hf_models.py fetch  --repo-id USER/nile-wallet-classifiers              # into results/runs/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
MODELS = ("e5-small-e15", "route-e5small")  # run names; weights live in <run>/model


def model_dir(name: str) -> Path:
    return RUNS / name / "model"


def check_local(name: str) -> Path:
    d = model_dir(name)
    if not d.is_dir():
        raise SystemExit(f"{d} is missing")
    if not (d / "config.json").is_file():
        raise SystemExit(f"{d} has no config.json")
    if not (any(d.glob("*.safetensors")) or any(d.glob("*.bin"))):
        raise SystemExit(f"{d} has no weights file (*.safetensors or *.bin)")
    return d


def upload(repo_id: str, public: bool) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN to a Hugging Face write token (never commit it)")
    folders = [(name, check_local(name)) for name in MODELS]  # fail before any network call
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    for name, folder in folders:
        print(f"uploading {name} ({folder}) ...", flush=True)
        api.upload_folder(folder_path=str(folder), path_in_repo=f"{name}/model", repo_id=repo_id,
                          repo_type="model", commit_message=f"Add {name}")
    print(f"done: https://huggingface.co/{repo_id}")


def fetch(repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(RUNS),
                      allow_patterns=[f"{name}/model/*" for name in MODELS],
                      token=os.getenv("HF_TOKEN"))
    for name in MODELS:
        print(f"ok: {check_local(name)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--repo-id", required=True)
    up.add_argument("--public", action="store_true", help="default is a private repo")
    fe = sub.add_parser("fetch")
    fe.add_argument("--repo-id", required=True)
    args = p.parse_args()
    if args.cmd == "upload":
        upload(args.repo_id, args.public)
    else:
        fetch(args.repo_id)


if __name__ == "__main__":
    main()