"""Shared test setup: saved studies always go to a temporary folder, never data/cache/."""

from pathlib import Path

import pytest

from backend.agent import study_store


@pytest.fixture(autouse=True)
def isolated_study_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect data/cache/studies to a per-test temporary folder."""
    folder = tmp_path / "studies"
    monkeypatch.setattr(study_store, "studies_dir", lambda: folder)
    return folder


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never send LangSmith traces, even when .env turns tracing on (no key needed)."""
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")


@pytest.fixture(autouse=True)
def no_live_weather(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call NASA POWER: the climate is unknown unless a test stubs it.

    backend/sun.py's own tests stub the HTTP layer themselves, so they are left alone.
    """
    if request.node.path.name == "test_sun.py":
        return
    from backend import sun

    monkeypatch.setattr(sun, "_fetch_power", lambda lat, lon: None)
