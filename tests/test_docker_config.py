from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


def test_api_healthcheck_has_start_period():
    hc = load_compose()["services"]["api"]["healthcheck"]
    assert hc["test"][0] == "CMD"
    assert "start_period" in hc


def test_db_and_model_mounts_are_read_only():
    volumes = load_compose()["services"]["api"]["volumes"]
    for target in ("/app/data/db", "results/runs/route-e5small/model", "results/runs/e5-small-e15/model"):
        mounts = [v for v in volumes if target in v]
        assert len(mounts) == 1, target
        assert mounts[0].endswith(":ro"), mounts[0]


def test_no_secrets_baked_in():
    api = load_compose()["services"]["api"]
    env_file = api["env_file"]
    assert ".env" in ([env_file] if isinstance(env_file, str) else env_file)
    for name in ("Dockerfile", "compose.yaml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "API_KEY" not in text, name
    assert "COPY . " not in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerignore_excludes_secrets_and_weights():
    lines = {ln.strip() for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", "results", "venv", "data/db"} <= lines


def test_dockerfile_serves_on_all_interfaces():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "src/serve.py" in text and "0.0.0.0" in text


def test_requirements_pin_service_dependencies():
    lines = [ln.strip().lower() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    for pkg in ("fastapi", "uvicorn"):
        assert any(ln.startswith(pkg + "==") for ln in lines), f"{pkg} is not pinned"