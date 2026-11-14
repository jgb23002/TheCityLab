"""Smoke tests for the CityLab health endpoint and integrated static frontend."""

from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    """The health endpoint returns its expected JSON response."""
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_frontend() -> None:
    """The root mount serves the supplied CityLab interface and its static files."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>CityLab: field reports for buildings</title>" in response.text
        assert 'id="ask-form"' in response.text
        assert 'id="study-question"' in response.text
        assert 'id="demo-mode"' in response.text  # replaces the old Data select
        assert 'id="solar-ranking"' in response.text
        for element in ("progress", "cancel", "citation-badge", "error-banner", "cards", "ledger",
                        "ledger-filter", "chart-summary", "brief"):
            assert f'id="{element}"' in response.text, element
        script = client.get("/app.js")
        assert script.status_code == 200 and "/api/ask-stream" in script.text and "getReader" in script.text
        assert client.get("/style.css").status_code == 200
