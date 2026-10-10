import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_models", ROOT / "scripts" / "hf_models.py")
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)


def test_layout_matches_agent_constants():
    text = (ROOT / "src" / "agent" / "models.py").read_text(encoding="utf-8")
    for name in hf.MODELS:
        assert name in text
        assert hf.model_dir(name) == ROOT / "results" / "runs" / name / "model"


def test_layout_matches_compose_mounts():
    volumes = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]["api"]["volumes"]
    for name in hf.MODELS:
        assert f"./results/runs/{name}/model:/app/results/runs/{name}/model:ro" in volumes


def test_check_local_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "RUNS", tmp_path)
    with pytest.raises(SystemExit):
        hf.check_local("m")                       # folder missing
    d = tmp_path / "m" / "model"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit):
        hf.check_local("m")                       # no config.json
    (d / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        hf.check_local("m")                       # no weights
    (d / "model.safetensors").write_bytes(b"x")
    assert hf.check_local("m") == d


def test_upload_requires_token_before_any_network(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        hf.upload("user/repo", public=False)


def test_upload_checks_local_files_first(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(hf, "RUNS", tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        hf.upload("user/repo", public=False)