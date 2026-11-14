"""Smoke tests for the CityLab scaffold."""

from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    """The health endpoint returns its expected JSON response."""
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_frontend() -> None:
    """The root mount serves the CityLab placeholder and its static files."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>CityLab</title>" in response.text
        assert "CityLab is running" in response.text
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
