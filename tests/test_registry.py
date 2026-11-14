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
        "solar_payback", "rank_buildings", "draft_email", "lookup_property", "interior_inspect",
        "screen_upgrades", "read_evidence", "list_evidence",
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


# ---- retrofit: solar_payback, rank_buildings, draft_email -------------------------------

def retrofit_tools(ledger: Ledger, top_n: int = 10) -> dict:
    """Mock-mode retrofit tools with the LA cohort sampled and inspected."""
    tools = {t.name: t for t in registry.get_tools("mock", ledger, intent="retrofit", top_n=top_n)}
    sample = tools["sample_buildings"].invoke({"market_id": 20, "property_type": 11})
    for b in sample["buildings"]:
        tools["vision_inspect"].invoke({"property_id": b["id"]})
    tools["_ids"] = [b["id"] for b in sample["buildings"]]
    return tools


def test_retrofit_sampling_needs_no_regression_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrofit keeps buildings without rent or controls (it only needs a roof)."""
    rows = [complete(1), complete(2, max_rent=None, year_built=None, building_class=0)]
    monkeypatch.setattr(costar_data, "find_buildings", lambda market_id, property_type, limit=20: rows)
    tools = {t.name: t for t in registry.get_tools("real", Ledger(), question="parking and rent", intent="retrofit")}
    assert [b["id"] for b in tools["sample_buildings"].invoke({"market_id": 20, "property_type": 11})["buildings"]] == [1, 2]


def test_solar_rank_and_draft_are_wired_and_recorded() -> None:
    """Each building is estimated from ledger records, ranked once, then drafted from the top."""
    ledger = Ledger()
    tools = retrofit_tools(ledger)
    for pid in tools["_ids"]:
        est = tools["solar_payback"].invoke({"property_id": pid})
        assert est["source"] == "mock" and est["property_id"] == pid
    ranked = tools["rank_buildings"].invoke({})
    assert ranked["estimated"] == 12 and ranked["skipped_no_roof"] == 0
    assert [r["rank"] for r in ranked["ranked"]] == list(range(1, 13))
    paybacks = [r["payback_years"] for r in ranked["ranked"]]
    assert paybacks == sorted(paybacks)
    top = ranked["ranked"][0]["property_id"]
    draft = tools["draft_email"].invoke({"property_id": top})
    assert draft["property_id"] == top and draft["body_markdown"]
    entry = ledger.get(draft["evidence_id"])
    assert entry["tool"] == "draft_email" and entry["args"]["rank"] == 1
    assert registry.ranking_ok(ledger.get(ranked["evidence_id"])["result"])
    assert registry.draft_ok(entry["result"])


def test_solar_uses_measured_roof_never_floor_area_and_skips_missing_roof() -> None:
    """A building with no measured roof is skipped (and counted), even if it has floor area."""
    from backend.schemas import Measurement

    ledger = Ledger()
    tools = retrofit_tools(ledger)
    first, second = tools["_ids"][:2]
    # Overwrite the first building's measurement with one that has no roof area.
    ledger.add("vision_inspect", {"property_id": first},
               Measurement(id="m", property_id=first, roof_area_sq_ft=None).model_dump(mode="json"), "mock")
    skipped = tools["solar_payback"].invoke({"property_id": first})
    assert skipped["skipped"] is True and "roof" in skipped["reason"]
    est = tools["solar_payback"].invoke({"property_id": second})
    assert ledger.get(est["evidence_id"])["args"]["roof_area_sq_ft"] > 0
    ranked = tools["rank_buildings"].invoke({})
    assert ranked["estimated"] == 1 and ranked["skipped_no_roof"] == 1
    assert [r["property_id"] for r in ranked["ranked"]] == [second]


def test_solar_needs_an_inspected_building_and_rank_needs_estimates() -> None:
    """Missing inputs fail loudly with ledger errors."""
    ledger = Ledger()
    tools = {t.name: t for t in registry.get_tools("mock", ledger, intent="retrofit")}
    assert "solar_payback failed" in tools["solar_payback"].invoke({"property_id": 990101})
    assert "rank_buildings failed" in tools["rank_buildings"].invoke({})
    assert all("error" in e["result"] for e in ledger.all())


def test_draft_email_only_for_top_n() -> None:
    """Drafts are limited to the top N ranked buildings."""
    ledger = Ledger()
    tools = retrofit_tools(ledger, top_n=2)
    for pid in tools["_ids"]:
        tools["solar_payback"].invoke({"property_id": pid})
    ranked = tools["rank_buildings"].invoke({})["ranked"]
    assert tools["draft_email"].invoke({"property_id": ranked[1]["property_id"]})["evidence_id"]
    out = tools["draft_email"].invoke({"property_id": ranked[2]["property_id"]})
    assert "draft_email failed" in out and "top 2" in out


def test_real_mode_solar_reuses_the_vision_cache(real_vision: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second retrofit run makes no new vision model call; solar runs on the cached measurement."""
    from backend.schemas import Measurement, SolarResult
    from backend.tools import solar

    real_vision["observations"] = {"notes": "ok"}
    monkeypatch.setattr(solar, "estimate_solar", lambda building, vision_result: SolarResult(
        id=f"s{building.id}", property_id=building.id, payback_years=7.0,
        usable_roof_area_sq_ft=vision_result.roof_area_sq_ft))
    from backend.tools import vision
    monkeypatch.setattr(vision, "measure_building", lambda image_path, property_id: (
        real_vision.__setitem__("calls", real_vision["calls"] + 1) or
        Measurement(id=f"m-{property_id}", property_id=property_id, roof_area_sq_ft=5000.0)))
    for _ in range(2):
        ledger = Ledger()
        tools = {t.name: t for t in registry.get_tools("real", ledger, intent="retrofit")}
        tools["sample_buildings"].invoke({"market_id": 20, "property_type": 11})
        tools["vision_inspect"].invoke({"property_id": 7})
        assert tools["solar_payback"].invoke({"property_id": 7})["usable_roof_area_sq_ft"] == 5000.0
    assert real_vision["calls"] == 1


@pytest.mark.parametrize("question, intent", [
    ("Which 10 Los Angeles apartment buildings are the best rooftop-solar candidates?", "retrofit"),
    ("Which roofs could hold solar panels?", "retrofit"),
    ("Plan a retrofit for Houston", "retrofit"),
    ("Show the best candidates in LA", "retrofit"),
    ("In Houston, do apartments with more parking have higher rent?", "research"),
])
def test_detect_intent(question: str, intent: str) -> None:
    """auto picks retrofit for solar/rooftop/retrofit/'best candidates', else research."""
    assert registry.detect_intent(question) == intent
    assert registry.detect_intent(question, "research") == "research"


def test_parse_top_n() -> None:
    """N is the first non-year number in the question, default 10, at most 10."""
    assert registry.parse_top_n("Which 5 buildings?") == 5
    assert registry.parse_top_n("Top 25 since 2020") == 10
    assert registry.parse_top_n("Best roofs built after 1990?") == 10
