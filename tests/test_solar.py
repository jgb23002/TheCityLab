"""Solar estimate tests; PVWatts is never called."""

import pytest

from backend.schemas import Building, Measurement, SolarResult
from backend.tools import geometry, solar


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(solar.config, "NREL_API_KEY", None)
    monkeypatch.setattr(solar.config, "CACHE_DIR", tmp_path)


def test_fallback_payback_math_for_houston() -> None:
    b = Building(id=1, market_id=14, latitude=29.7, longitude=-95.4, name="Test Tower",
                 gross_typical_floor_sq_ft=10_000)
    m = Measurement(id="m-1", property_id=1, observations={"rooftop_obstructions": "few"})
    r = solar.estimate_solar(b, m)
    assert r.usable_roof_area_sq_ft == 6000
    assert r.capacity_kw == 72.0
    assert r.annual_energy_kwh == 72 * 1300
    a = r.assumptions
    assert a["estimated_cost_usd"] == 144_000
    assert a["estimated_annual_savings_usd"] == round(72 * 1300 * 0.09)
    assert r.score == a["estimated_payback_years"] == round(144_000 / (72 * 1300 * 0.09), 1)
    assert r.payback_years == r.score
    assert any("regional average" in c for c in r.caveats)


def test_estimate_solar_uses_parcel_bearing_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_pvwatts(lat, lon, system_kw, azimuth_deg=solar.PANEL_AZIMUTH_DEG):
        captured["azimuth"] = azimuth_deg
        return {"ac_annual": 50_000.0, "ac_monthly": None, "station_info": None}

    monkeypatch.setattr(solar, "pvwatts_result", fake_pvwatts)
    b = Building(id=15, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000)
    m = Measurement(id="m-15", property_id=15, observations={"parcel_bearing_deg": 37.5})
    solar.estimate_solar(b, m)
    assert captured["azimuth"] == 37.5


def test_estimate_solar_falls_back_to_due_south_without_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_pvwatts(lat, lon, system_kw, azimuth_deg=solar.PANEL_AZIMUTH_DEG):
        captured["azimuth"] = azimuth_deg
        return {"ac_annual": 50_000.0, "ac_monthly": None, "station_info": None}

    monkeypatch.setattr(solar, "pvwatts_result", fake_pvwatts)
    b = Building(id=16, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000)
    r = solar.estimate_solar(b)
    assert captured["azimuth"] == solar.PANEL_AZIMUTH_DEG
    assert r.assumptions["azimuth_source"].startswith("assumed due south")


def test_estimate_solar_prefers_vexcel_measured_roof_over_costar_math() -> None:
    b = Building(id=17, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=99_999.0)
    m = Measurement(id="m-17", property_id=17, roof_area_sq_ft=12_345.0,
                     observations={"source": "Vexcel precomputed parcel attributes"})
    r = solar.estimate_solar(b, m)
    assert r.assumptions["estimated_roof_area_sq_ft"] == 12_345.0
    assert r.assumptions["roof_area_source"] == "Vexcel detected structure footprint"


def test_estimate_solar_ignores_unlabeled_roof_area_and_uses_costar_math() -> None:
    """A roof_area_sq_ft without the Vexcel-measurement source tag is a vision reading, not used."""
    b = Building(id=18, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000.0)
    m = Measurement(id="m-18", property_id=18, roof_area_sq_ft=99_999.0, observations={})
    r = solar.estimate_solar(b, m)
    assert r.assumptions["estimated_roof_area_sq_ft"] == 10_000.0
    assert r.assumptions["roof_area_source"].startswith("CoStar building footprint")


def test_estimate_solar_runs_without_any_vision_result() -> None:
    """Vision is optional enrichment only; the estimate must not require it."""
    b = Building(id=14, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000)
    r = solar.estimate_solar(b)
    assert r.capacity_kw is not None
    assert r.usable_roof_area_sq_ft == 6000  # default "none" obstruction fraction (0.60) applies


def test_estimate_solar_reports_matterport_height_when_available() -> None:
    b = Building(id=19, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000,
                 num_stories_above_grade=2)
    m = Measurement(id="m-19", property_id=19, observations={"matterport_building_height_ft": 9.1})
    r = solar.estimate_solar(b, m)
    assert r.assumptions["estimated_building_height_ft"] == 9.1
    assert r.assumptions["building_height_source"] == "Matterport room scan (median room height)"


def test_estimate_solar_falls_back_to_costar_height_without_matterport() -> None:
    b = Building(id=20, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000,
                 num_stories_above_grade=2)
    r = solar.estimate_solar(b)
    assert r.assumptions["estimated_building_height_ft"] == 2 * geometry.FLOOR_TO_FLOOR_FT
    assert r.assumptions["building_height_source"].startswith("CoStar stories")


def test_pvwatts_result_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solar, "pvwatts_result", lambda lat, lon, kw, azimuth_deg=solar.PANEL_AZIMUTH_DEG: {
        "ac_annual": 100_000.0, "ac_monthly": [8_000.0] * 12,
        "station_info": {"lat": 34.05, "lon": -118.25, "distance": 1200, "elev": 71},
    })
    r = solar.estimate_solar(Building(id=2, market_id=20, latitude=34, longitude=-118,
                                      gross_typical_floor_sq_ft=10_000))
    assert r.annual_energy_kwh == 100_000
    assert r.assumptions["energy_source"] == "NREL PVWatts v8"
    assert r.assumptions["ac_monthly_kwh"] == [8_000.0] * 12
    assert r.assumptions["weather_station"]["distance"] == 1200


def test_missing_roof_gives_no_estimate() -> None:
    """No footprint fields on the Building (GBA alone, with no stories to divide by) means no estimate."""
    r = solar.estimate_solar(Building(id=3, building_size_sq_ft=50_000), Measurement(id="m-3"))
    assert r.capacity_kw is None and r.score is None and r.caveats


def test_rank_by_payback_with_unknowns_last() -> None:
    ranked = solar.rank_buildings([
        SolarResult(id="a", score=9.0), SolarResult(id="b"), SolarResult(id="c", score=5.0),
    ])
    assert [(r.id, r.rank) for r in ranked] == [("c", 1), ("a", 2), ("b", None)]


def test_rank_accepts_shared_payback_field_and_preserves_legacy_scores() -> None:
    """Planner-shaped estimates work alongside the AI tool's earlier score format."""
    inputs = [
        SolarResult(id="unknown"), SolarResult(id="legacy", score=8),
        SolarResult(id="explicit", payback_years=5, score=100),
    ]
    ranked = solar.rank_buildings(inputs)
    assert [(r.id, r.rank) for r in ranked] == [("explicit", 1), ("legacy", 2), ("unknown", None)]
    assert all(result.rank is None for result in inputs)
