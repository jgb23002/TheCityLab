"""Registry: tool modes, auto fallback, ledger ids and recorded sources."""

import logging

import pytest

from backend.agent import registry
from backend.agent.evidence import Ledger
from backend.schemas import Building
from backend.tools import costar_data


def complete(id: int, **extra: object) -> Building:
    """A Building with every field the regression needs (units, year, class, rent)."""
    fields = dict(id=id, market_id=14, property_type=11, latitude=29.7, longitude=-95.4,
                  total_num_units=100, year_built=1990, building_class=2, max_rent=1500.0,
                  vacancy_percent=5.0)
    fields.update(extra)
    return Building(**fields)


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
        return [complete(7, market_id=market_id, property_type=property_type)]

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
    # Only the one inspected building enters the regression (joined by property_id).
    assert stats["sample_size"] == 1 and ledger.get("E5")["args"]["joined_rows"] == 1
    assert tools["read_evidence"].invoke({"evidence_id": "E5"})["tool"] == "run_regression"
    assert [row["id"] for row in tools["list_evidence"].invoke({})] == ["E1", "E2", "E3", "E4", "E5"]


def test_sample_size_defaults_to_n_and_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """limit defaults to the run's n (40) and is capped at 60; more candidates are scanned."""
    seen: list[int] = []

    def fake_real(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
        seen.append(limit)
        return []

    monkeypatch.setattr(costar_data, "find_buildings", fake_real)
    tools = tools_for("real", Ledger())
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11, "limit": 500})
    assert seen == [200, 300]  # limit x 5 candidates scanned for complete cases
    assert registry.get_tools("real", Ledger(), n=999)  # n is capped, not rejected


def test_sampling_keeps_only_complete_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buildings missing a control or the question's outcome are excluded and counted."""
    rows = [complete(1), complete(2, max_rent=None), complete(3, year_built=None),
            complete(4, building_class=0), complete(5), complete(6)]
    monkeypatch.setattr(costar_data, "find_buildings", lambda market_id, property_type, limit=20: rows)
    ledger = Ledger()
    tools = {t.name: t for t in registry.get_tools("real", ledger, n=2, question="parking and rent")}
    out = tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    assert [b["id"] for b in out["buildings"]] == [1, 5]
    assert out["count"] == 2 and out["excluded_incomplete"] == 3 and out["candidates_scanned"] == 6
    assert ledger.get("E1")["args"]["required_fields"][-1] == "max_rent"
    # Without a rent question, rent is not required.
    tools = {t.name: t for t in registry.get_tools("real", Ledger(), n=3, question="vacancy and paving")}
    assert [b["id"] for b in tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})["buildings"]] == [1, 2, 5]


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
        return [complete(7), complete(8)]

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
    ledger2 = Ledger()
    tools2 = tools_for("real", ledger2)
    tools2["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    second = tools2["vision_inspect"].invoke({"property_id": 7})
    assert real_vision["calls"] == 1
    assert [e["tool"] for e in ledger2.all()] == ["sample_buildings", "get_aerial", "vision_inspect"]
    assert ledger2.get(second["evidence_id"])["args"]["image_path"] == "img.jpg"
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
    tools = tools_for("real", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    out = tools["vision_inspect"].invoke({"property_id": 7})
    assert "CacheMiss" in out and real_vision["calls"] == 0
    assert ledger.all()[-1]["tool"] == "vision_inspect" and "error" in ledger.all()[-1]["result"]


# ---- image resolution, regression join and idempotence ---------------------------------

def test_empty_image_path_triggers_get_aerial_first(real_vision: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty image_path fetches the aerial image (own ledger entry) and passes its path on."""
    from backend.tools import vision

    seen: list[str] = []
    original = vision.measure_building

    def spy(image_path: str, property_id: int):
        seen.append(image_path)
        return original(image_path, property_id)

    monkeypatch.setattr(vision, "measure_building", spy)
    ledger = Ledger()
    tools = tools_for("real", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    tools["vision_inspect"].invoke({"property_id": 7, "image_path": ""})
    assert seen == ["img.jpg"]
    assert [e["tool"] for e in ledger.all()] == ["sample_buildings", "get_aerial", "vision_inspect"]
    assert ledger.get("E3")["args"]["image_path"] == "img.jpg"


def test_missing_image_fails_loudly(real_vision: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """If get_aerial yields no image, a ledger error is recorded and vision is never called."""
    from backend.schemas import AerialImage
    from backend.tools import vexcel

    monkeypatch.setattr(vexcel, "get_aerial", lambda lat, lon, property_id: AerialImage(id="a", image_path=None))
    ledger = Ledger()
    tools = tools_for("real", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    out = tools["vision_inspect"].invoke({"property_id": 7})
    assert "vision_inspect failed" in out and "returned no image" in out
    assert real_vision["calls"] == 0
    assert "error" in ledger.all()[-1]["result"]
    # A property that was never sampled cannot get an image either.
    out = tools["vision_inspect"].invoke({"property_id": 999})
    assert "get_aerial failed" in out and real_vision["calls"] == 0


def test_regression_joins_inspected_properties_and_ignores_model_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_regression receives exactly the inspected buildings + measurements, by property_id."""
    from backend.schemas import Measurement, StatsResult
    from backend.tools import stats

    captured: dict = {}

    def fake_study(question, buildings, measurements=None):
        captured["b"] = [b.id for b in buildings]
        captured["m"] = [m.property_id for m in measurements or []]
        captured["rent"] = [b.max_rent for b in buildings]
        return StatsResult(id="s", outcome="max_rent", predictor="parking_space_count",
                           coefficients={"parking_space_count": 2.5}, sample_size=len(buildings))

    rows = [complete(i, max_rent=1000.0 + i) for i in (1, 2, 3)]
    monkeypatch.setattr(costar_data, "find_buildings", lambda market_id, property_type, limit=20: rows)
    monkeypatch.setattr(stats, "run_study", fake_study)
    ledger = Ledger()
    tools = tools_for("real", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    # Inspect 3 then 1 (not 2), plus a stray measurement for an unsampled property.
    for pid in (3, 1, 42):
        ledger.add("vision_inspect", {"property_id": pid},
                   Measurement(id=f"m{pid}", property_id=pid, parking_space_count=10).model_dump(mode="json"), "real")
    thin = [{"id": 1}, {"id": 2}]  # what an LLM might pass: no rent, no controls
    out = tools["run_regression"].invoke({"question": "parking and rent", "buildings": thin})
    assert captured == {"b": [1, 3], "m": [1, 3], "rent": [1001.0, 1003.0]}
    args = ledger.get(out["evidence_id"])["args"]
    assert args["joined_rows"] == 2 and args["inspected_properties"] == 3
    assert args["sampled_buildings"] == 3 and args["model_supplied_rows_ignored"] == 2


def test_second_regression_returns_the_first_entry() -> None:
    """A repeat run_regression for the same outcome/predictor reuses the first evidence id."""
    ledger = Ledger()
    tools = tools_for("mock", ledger)
    b = tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})["buildings"][0]
    tools["vision_inspect"].invoke({"property_id": b["id"]})
    first = tools["run_regression"].invoke({"question": "Do more parking spaces mean higher rent?"})
    count = len(ledger.all())
    second = tools["run_regression"].invoke({"question": "Is rent associated with parking (reworded)?"})
    assert second["evidence_id"] == first["evidence_id"] and second["reused"] is True
    assert len(ledger.all()) == count  # no new entry
    other = tools["run_regression"].invoke({"question": "Is vacancy associated with paved surface?"})
    assert other["evidence_id"] != first["evidence_id"]


def test_failed_regression_without_rows_is_a_ledger_error() -> None:
    """No inspected properties means no regression input: fail loudly in the ledger."""
    ledger = Ledger()
    tools = tools_for("mock", ledger)
    tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
    out = tools["run_regression"].invoke({"question": "parking and rent"})
    assert "run_regression failed" in out and "error" in ledger.all()[-1]["result"]


def test_regression_ok_requires_coefficient_and_sample_size() -> None:
    """A StatsResult with findings.error or n=0 is not a result."""
    assert registry.regression_ok({"predictor": "p", "coefficients": {"p": 1.0}, "sample_size": 12})
    assert not registry.regression_ok({"predictor": "p", "coefficients": None, "sample_size": 0,
                                       "findings": {"error": "not enough complete rows (n=0)"}})
    assert not registry.regression_ok({"predictor": "p", "coefficients": {"p": 1.0}, "sample_size": 0})
    assert not registry.regression_ok({"error": "boom"})
