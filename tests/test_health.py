"""Smoke tests for the CityLab health endpoint and integrated static frontend."""

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.main import app


def test_health() -> None:
    """The health endpoint returns its expected JSON response."""
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_config_exposes_only_the_publishable_sdk_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/config returns the Matterport Showcase SDK key only, never the server token pair."""
    monkeypatch.setattr(main, "MP_SDK_KEY", "pk_test_123")
    with TestClient(app) as client:
        response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {"matterport_sdk_key": "pk_test_123"}
    for leaked in ("MP_TOKEN_ID", "MP_TOKEN_SECRET", "token"):
        assert leaked not in response.text


def test_home_page_is_the_solar_page() -> None:
    """The root serves the solar page and its relative assets; /solar/ still works."""
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200 and "<title>CityLab Solar</title>" in home.text
        for asset in ("/solar.css", "/flight.js", "/logo.svg", "/data_houston.json"):
            assert client.get(asset).status_code == 200, asset
        script = client.get("/solar.js")
        assert script.status_code == 200 and "/api/solar-city/stream" in script.text
        assert client.get("/solar/").text == home.text
        assert client.get("/solar/solar.js").status_code == 200


def test_frontend() -> None:
    """/classic serves the original question-box interface and its static files."""
    with TestClient(app) as client:
        response = client.get("/classic")
        assert response.status_code == 200
        assert "<title>CityLab: field reports for buildings</title>" in response.text
        assert 'id="ask-form"' in response.text
        assert 'id="study-question"' in response.text
        assert 'id="demo-mode"' in response.text  # replaces the old Data select
        assert 'id="solar-ranking"' in response.text
        for element in ("progress", "cancel", "citation-badge", "error-banner", "cards", "ledger",
                        "ledger-filter", "chart-summary", "brief", "strip"):
            assert f'id="{element}"' in response.text, element
        script = client.get("/app.js")
        assert script.status_code == 200 and "/api/ask-stream" in script.text and "getReader" in script.text
        assert client.get("/style.css").status_code == 200


def test_showcase_strip_returns_404_with_no_qualifying_building(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Nothing in an empty cache has a measurement, boxed photo and roof report yet: 404, not an error."""
    monkeypatch.setattr(main, "CACHE_DIR", tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/showcase-strip")
    assert response.status_code == 404


def test_showcase_strip_picks_the_first_fully_measured_building(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A building with a cached measurement, boxed photo and roof report is returned, enriched and costed."""
    from backend.schemas import Building, Measurement

    building = Building(id=777, market_id=14, property_type=11, latitude=29.7, longitude=-95.4,
                        name="Test Towers", gross_typical_floor_sq_ft=10_000.0, num_stories_above_grade=2)
    monkeypatch.setattr(main, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(main, "sample_buildings", lambda city, property_type, n=50: [building])
    monkeypatch.setattr(main.costar_data, "get_property", lambda id: building)
    (tmp_path / "vision").mkdir()
    (tmp_path / "vision" / "m-777.json").write_text(
        Measurement(id="m-777", property_id=777).model_dump_json(), encoding="utf-8")
    (tmp_path / "vexcel" / "detect").mkdir(parents=True)
    (tmp_path / "vexcel" / "detect" / "property_777_boxes.jpg").write_bytes(b"jpg")
    (tmp_path / "roof_view").mkdir()
    (tmp_path / "roof_view" / "property_777_report.jpg").write_bytes(b"jpg")
    with TestClient(app) as client:
        response = client.get("/api/showcase-strip")
    assert response.status_code == 200
    body = response.json()
    assert body["property_id"] == 777 and body["name"] == "Test Towers"
    assert body["roof_area_sq_ft"] == 10_000.0 and body["payback_years"] is not None
