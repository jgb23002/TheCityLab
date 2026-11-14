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
        assert 'id="data-source"' in response.text
        assert 'id="solar-ranking"' in response.text
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
