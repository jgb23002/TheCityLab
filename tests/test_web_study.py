"""Exercise the real web orchestration with fake imagery and the real stats/solar tools."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agent import study
from backend.schemas import AskResponse
from backend.tools import mock


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Supply fixed test data while leaving actual regression/solar code in use."""
    buildings = mock.find_buildings(14, 11, 12)
    buildings[0].vacancy_percent = 0
    monkeypatch.setattr(study.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(study.config, "COSTAR_DATA_DIR", tmp_path / "kit")
    monkeypatch.setattr(study.config, "USE_CACHE_ONLY", False)
    monkeypatch.setattr(study.config, "NREL_API_KEY", None)
    monkeypatch.setattr(study.config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(study.config, "VISION_MODEL", "test-vision-model")
    sampler = Mock(return_value=buildings)
    monkeypatch.setattr(study.samples, "sample_buildings", sampler)
    aerial = Mock(side_effect=mock.get_aerial)
    parcel = Mock(side_effect=mock.get_property_info)
    monkeypatch.setattr(study.vexcel, "get_aerial", aerial)
    monkeypatch.setattr(study.vexcel, "get_property_info", parcel)

    def measurement(path: str, property_id: int):
        """Return fake image evidence; it is never presented as a production observation."""
        return mock.measure_building(path, property_id).model_copy(update={"observations": {"basis": "test fixture"}})

    inspect = Mock(side_effect=measurement)
    monkeypatch.setattr(study.vision, "measure_building", inspect)
    regression = Mock(wraps=study.stats.run_study)
    monkeypatch.setattr(study.stats, "run_study", regression)
    return {"buildings": buildings, "sampler": sampler, "aerial": aerial,
            "parcel": parcel, "vision": inspect, "regression": regression}


def test_real_research_uses_merged_tools_complete_controls_and_cached_measurements(workflow: dict) -> None:
    """Q1/Q2 keep full records and reuse image results across different outcomes."""
    client = TestClient(main.app)
    rent = client.post("/api/ask", json={"question": "Do greener Houston apartments have higher rent?", "limit": 12})
    assert rent.status_code == 200
    result = AskResponse.model_validate(rent.json())
    assert not result.study.is_mock
    assert result.study.stats.sample_size == 12
    assert result.study.stats.outcome == "max_rent"
    assert result.study.stats.predictor == "green_cover_percent"
    assert result.study.stats.confidence_intervals["green_cover_percent"]
    passed = workflow["regression"].call_args.args[1]
    assert all(building.total_num_units and building.year_built and building.building_class for building in passed)
    assert workflow["aerial"].call_count == workflow["parcel"].call_count == workflow["vision"].call_count == 12

    vacancy = client.post("/api/ask", json={"question": "Does more paved surface mean higher vacancy?", "limit": 12})
    assert vacancy.status_code == 200
    changed = AskResponse.model_validate(vacancy.json())
    assert changed.study.stats.outcome == "vacancy_percent"
    assert changed.study.stats.predictor == "paved_surface_percent"
    assert changed.study.buildings[0].vacancy_percent == 0
    assert workflow["vision"].call_count == 12
    assert all(event.payload["cached"] for event in changed.events if event.event_type == "measure")


def test_real_data_enriches_cohort_through_costar_loader(workflow: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The merged CoStar tool supplies building names and other full-kit fields."""
    directory = study.config.COSTAR_DATA_DIR
    directory.mkdir()
    (directory / "commercial-property.jsonl").write_text("")
    enriched = workflow["buildings"][0].model_copy(update={"name": "CoStar enrichment test"})
    loader = Mock(return_value=[enriched, *workflow["buildings"][1:]])
    monkeypatch.setattr(study.costar_data, "load_properties", loader)
    study._kit_properties.cache_clear()
    response = TestClient(main.app).post("/api/ask", json={"question": "green cover and rent", "limit": 12})
    assert response.status_code == 200
    assert response.json()["study"]["buildings"][0]["name"] == "CoStar enrichment test"
    loader.assert_called_once()
    study._kit_properties.cache_clear()


def test_cache_only_miss_makes_no_network_or_model_calls(workflow: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing real cache produces a clear error instead of fake results or live calls."""
    monkeypatch.setattr(study.config, "USE_CACHE_ONLY", True)
    response = TestClient(main.app).post("/api/ask", json={"question": "green cover and rent", "limit": 12})
    assert response.status_code == 503
    workflow["aerial"].assert_not_called()
    workflow["vision"].assert_not_called()
    assert "No usable aerial measurements" in response.json()["detail"]


def test_partial_cache_keeps_missing_values_unknown(workflow: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported zero remains valid; uncached buildings cannot gain invented measurements."""
    building = workflow["buildings"][0]
    measurement = mock.measure_building("test.jpg", building.id).model_copy(update={
        "green_cover_percent": 0, "observations": {"basis": "test fixture"},
    })
    path = study.config.CACHE_DIR / "vision" / f"m-{building.id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(measurement.model_dump_json())
    monkeypatch.setattr(study.config, "USE_CACHE_ONLY", True)
    response = TestClient(main.app).post("/api/ask", json={"question": "green cover and rent", "limit": 12})
    assert response.status_code == 200
    result = AskResponse.model_validate(response.json())
    assert result.study.measurements[0].green_cover_percent == 0
    assert all(measurement.green_cover_percent is None for measurement in result.study.measurements[1:])
    assert result.study.stats.coefficients is None
    assert any("11 buildings lack" in caveat for caveat in result.study.caveats)
    workflow["aerial"].assert_not_called()


def test_solar_mode_returns_rankings_and_an_unsent_owner_draft(workflow: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI receives LA solar results, not Houston rent statistics."""
    buildings = mock.find_buildings(20, 11, 3)
    workflow["sampler"].return_value = buildings
    for building in buildings:
        measurement = mock.measure_building("test.jpg", building.id).model_copy(update={"observations": {"basis": "test fixture"}})
        path = study.config.CACHE_DIR / "vision" / f"m-{building.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(measurement.model_dump_json())
    monkeypatch.setattr(study.config, "USE_CACHE_ONLY", True)
    chat = Mock(side_effect=AssertionError("No live chat in cache-only mode"))
    monkeypatch.setattr(study.outreach.llm, "ask_text", chat)
    response = TestClient(main.app).post("/api/ask", json={
        "question": "Which Los Angeles buildings are best for solar?", "mode": "retrofit", "market_id": 20, "limit": 3,
    })
    assert response.status_code == 200
    result = AskResponse.model_validate(response.json())
    assert result.study.stats is None
    assert {building.market_id for building in result.study.buildings} == {20}
    assert [estimate.rank for estimate in result.study.solar_results] == [1, 2, 3]
    assert all(estimate.payback_years is not None for estimate in result.study.solar_results)
    assert result.study.outreach_drafts[0].body_markdown
    chat.assert_not_called()


def test_provider_error_does_not_expose_sdk_details(workflow: dict) -> None:
    """A failed provider request returns a useful public message without its raw body."""
    workflow["aerial"].side_effect = RuntimeError("private-sdk-sentinel")
    response = TestClient(main.app).post("/api/ask", json={"question": "green cover and rent"})
    assert response.status_code == 503
    assert "private-sdk-sentinel" not in response.text


def test_cohort_preview_shows_real_data_without_vision_calls(workflow: dict) -> None:
    """Users can see the merged data immediately without starting aerial analysis."""
    response = TestClient(main.app).get("/api/buildings?market_id=14&limit=12")
    assert response.status_code == 200
    assert len(response.json()) == 12
    assert response.json()[0]["vacancy_percent"] == 0
    workflow["aerial"].assert_not_called()
    workflow["vision"].assert_not_called()


def test_missing_vision_config_stops_before_imagery_calls(workflow: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete live configuration cannot start a large paid imagery run."""
    monkeypatch.setattr(study.config, "VISION_MODEL", None)
    response = TestClient(main.app).post("/api/ask", json={"question": "green cover and rent"})
    assert response.status_code == 503
    assert "VISION_MODEL" in response.json()["detail"]
    workflow["aerial"].assert_not_called()
