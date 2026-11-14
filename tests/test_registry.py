"""Registry: tool modes, auto fallback, ledger ids and recorded sources."""

import logging

import pytest

from backend.agent import registry
from backend.agent.evidence import Ledger
from backend.schemas import Building
from backend.tools import costar_data


def tools_for(mode: str, ledger: Ledger, n: int = 40) -> dict:
    """Return the registry tools by name."""
    return {t.name: t for t in registry.get_tools(mode, ledger, n=n)}


def not_ready(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
    """Stand in for a real tool that is not implemented yet."""
    raise NotImplementedError


def test_registry_exposes_contract_names() -> None:
    """All research, retrofit and evidence tools are registered."""
    assert set(tools_for("mock", Ledger())) == {
        "sample_buildings", "vexcel_detect", "vision_inspect", "run_regression",
        "solar_payback", "draft_email", "read_evidence", "list_evidence",
    }


def test_auto_falls_back_to_mock_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """NotImplementedError in auto mode uses the mock and logs one WARNING line."""
    monkeypatch.setattr(costar_data, "find_buildings", not_ready)
    ledger = Ledger()
    tools = tools_for("auto", ledger)
    with caplog.at_level(logging.WARNING, logger="backend.agent.registry"):
        first = tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
        tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11, "limit": 3})
    assert first["source"] == "mock" and first["count"] == 12
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["REAL TOOL NOT READY, USING MOCK: sample_buildings"]


def test_real_mode_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real mode never hides an unimplemented tool."""
    monkeypatch.setattr(costar_data, "find_buildings", not_ready)
    with pytest.raises(NotImplementedError):
        tools_for("real", Ledger())["sample_buildings"].invoke({"market_id": 14, "property_type": 11})


def test_real_mode_uses_real_function_and_records_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """A working real tool is called and its source is 'real'."""
    def fake_real(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
        return [Building(id=7, market_id=market_id, property_type=property_type, latitude=1.0, longitude=2.0)]

    monkeypatch.setattr(costar_data, "find_buildings", fake_real)
    ledger = Ledger()
    result = tools_for("real", ledger)["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    assert result["source"] == "real" and result["buildings"][0]["id"] == 7
    assert ledger.get("E1")["source"] == "real"


def test_ledger_ids_increment_and_results_carry_source() -> None:
    """Each call gets E1, E2, ...; results carry evidence_id and source."""
    ledger = Ledger()
    tools = tools_for("mock", ledger)
    sample = tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    b = sample["buildings"][0]
    detect = tools["vexcel_detect"].invoke({"lat": b["latitude"], "lon": b["longitude"]})
    inspect = tools["vision_inspect"].invoke({"property_id": b["id"]})
    stats = tools["run_regression"].invoke({"question": "parking and rent"})
    assert [sample["evidence_id"], detect["evidence_id"]] == ["E1", "E2"]
    assert inspect["evidence_id"] == "E4"  # E3 is the aerial image fetched for it
    assert [e["id"] for e in ledger.all()] == ["E1", "E2", "E3", "E4", "E5"]
    assert ledger.get("E3")["tool"] == "get_aerial"
    assert all(r["source"] == "mock" for r in (sample, detect, inspect, stats))
    assert stats["sample_size"] == 12 and ledger.get("E5")["args"]["measurement_count"] == 1
    assert tools["read_evidence"].invoke({"evidence_id": "E5"})["tool"] == "run_regression"
    assert [row["id"] for row in tools["list_evidence"].invoke({})] == ["E1", "E2", "E3", "E4", "E5"]


def test_sample_size_defaults_to_n_and_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """limit defaults to the run's n (40) and is capped at 60."""
    seen: list[int] = []

    def fake_real(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
        seen.append(limit)
        return []

    monkeypatch.setattr(costar_data, "find_buildings", fake_real)
    tools = tools_for("real", Ledger())
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11, "limit": 500})
    assert seen == [40, 60]
    assert registry.get_tools("real", Ledger(), n=999)  # n is capped, not rejected


def test_tool_errors_are_recorded_and_returned_to_the_model() -> None:
    """A failing tool records an error entry and returns its message instead of crashing."""
    ledger = Ledger()
    out = tools_for("mock", ledger)["vexcel_detect"].invoke({"lat": 0.0, "lon": 0.0})
    assert "vexcel_detect failed (E1)" in out
    assert "error" in ledger.get("E1")["result"]


def test_mode_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """TOOLS_MODE blank means auto; invalid values are rejected."""
    monkeypatch.setenv("TOOLS_MODE", "")
    assert registry.resolve_mode() == "auto"
    monkeypatch.setenv("TOOLS_MODE", " MOCK ")
    assert registry.resolve_mode() == "mock"
    assert registry.resolve_mode("real") == "real"
    with pytest.raises(ValueError):
        registry.resolve_mode("live")


# ---- vision measurement cache -------------------------------------------------------

@pytest.fixture
def real_vision(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    """Real-mode sample/aerial/vision stubs with the cache under tmp_path; counts model calls."""
    from backend import config
    from backend.schemas import AerialImage, Measurement
    from backend.tools import vexcel, vision

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    state = {"calls": 0, "observations": {"notes": "ok"}, "tmp": tmp_path}

    def find(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
        return [Building(id=7, market_id=14, property_type=11, latitude=29.7, longitude=-95.4)]

    def aerial(lat: float, lon: float, property_id: int) -> AerialImage:
        return AerialImage(id=f"aerial-{property_id}", property_id=property_id, image_path="img.jpg")

    def measure(image_path: str, property_id: int) -> Measurement:
        state["calls"] += 1
        return Measurement(id=f"m-{property_id}", property_id=property_id, parking_space_count=42,
                           observations=dict(state["observations"]))

    monkeypatch.setattr(costar_data, "find_buildings", find)
    monkeypatch.setattr(vexcel, "get_aerial", aerial)
    monkeypatch.setattr(vision, "measure_building", measure)
    return state


def test_second_vision_call_uses_cache_without_model_call(real_vision: dict) -> None:
    """A saved real measurement is reused: source real, cached true, no model call."""
    ledger = Ledger()
    tools = tools_for("real", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    first = tools["vision_inspect"].invoke({"property_id": 7})
    assert real_vision["calls"] == 1 and "cached" not in first
    assert (real_vision["tmp"] / "vision" / "m-7.json").exists()
    second = tools_for("real", Ledger())["vision_inspect"].invoke({"property_id": 7})
    assert real_vision["calls"] == 1
    assert second["cached"] is True and second["source"] == "real" and second["parking_space_count"] == 42


def test_vision_errors_are_not_cached(real_vision: dict) -> None:
    """A measurement with observations.error is returned but never saved."""
    real_vision["observations"] = {"error": "vision call failed: Timeout"}
    tools = tools_for("real", Ledger())
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    tools["vision_inspect"].invoke({"property_id": 7})
    tools["vision_inspect"].invoke({"property_id": 7})
    assert real_vision["calls"] == 2
    assert not (real_vision["tmp"] / "vision" / "m-7.json").exists()


def test_mock_measurements_are_never_cached(real_vision: dict) -> None:
    """Mock mode neither reads nor writes the real vision cache."""
    tools = tools_for("mock", Ledger())
    b = tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})["buildings"][0]
    tools["vision_inspect"].invoke({"property_id": b["id"]})
    assert not (real_vision["tmp"] / "vision").exists()


def test_cache_only_miss_is_an_error_without_model_call(
    real_vision: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USE_CACHE_ONLY with no cached measurement fails instead of calling the model."""
    from backend import config

    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    ledger = Ledger()
    out = tools_for("real", ledger)["vision_inspect"].invoke({"property_id": 7})
    assert "CacheMiss" in out and real_vision["calls"] == 0
    assert "error" in ledger.get("E1")["result"]
