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


def test_sun_context_reports_geometry_climate_and_plane_when_climate_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import sun

    monkeypatch.setattr(sun, "climate", lambda lat, lon: {
        "sunlight_kwh_m2_day": 4.65, "clear_sky_kwh_m2_day": 6.03, "cloud_cover_percent": 52.0,
        "cloud_loss_percent": 23.0, "source": "test climate"})
    monkeypatch.setattr(sun, "sky_year", lambda lat, lon, **kw: [
        {"month": 0, "elevation_deg": 45.0, "azimuth_deg": 180.0, "ghi": 800.0, "dhi": 200.0, "dni": 848.5, "hours": 10.0}])
    b = Building(id=21, market_id=14, latitude=29.76, longitude=-95.37, gross_typical_floor_sq_ft=10_000)
    s = solar.estimate_solar(b).assumptions["sun"]
    assert s["cloud_loss_percent"] == 23.0 and s["climate_source"] == "test climate"
    assert s["best_fixed_tilt_deg"] == round(0.76 * 29.76 + 3.1, 1)
    assert s["plane_kwh_m2_year"] > 0 and s["plane_facing_deg"] == solar.PANEL_AZIMUTH_DEG


def test_sun_context_without_climate_keeps_geometry_but_skips_the_plane_figure() -> None:
    b = Building(id=22, market_id=14, latitude=29.76, longitude=-95.37, gross_typical_floor_sq_ft=10_000)
    s = solar.estimate_solar(b).assumptions["sun"]
    assert s["best_panel_azimuth_deg"] == 180.0 and s["summer_solstice"]
    assert "plane_kwh_m2_year" not in s and "sunlight_kwh_m2_day" not in s


def test_owner_framing_uses_last_sale_price_and_units() -> None:
    b = Building(id=23, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000,
                 total_num_units=100, last_sale_price=12_000_000.0, last_sale_date="2024-05-01")
    a = solar.estimate_solar(b).assumptions
    assert a["system_cost_pct_of_last_sale_price"] == round(100 * a["estimated_cost_usd"] / 12_000_000, 1)
    assert a["savings_per_unit_per_month_usd"] == round(a["estimated_annual_savings_usd"] / 100 / 12, 2)
    assert a["last_sale_date"] == "2024-05-01"


def test_matterpak_cross_checks_pass_through_with_an_interior_caveat() -> None:
    b = Building(id=24, market_id=14, latitude=29.7, longitude=-95.4, gross_typical_floor_sq_ft=10_000)
    m = Measurement(id="m-24", property_id=24, observations={
        "matterpak_interior_height_ft": 9.6, "matterpak_interior_footprint_sq_ft": 8800.0})
    r = solar.estimate_solar(b, m)
    assert r.assumptions["matterpak_interior_height_ft"] == 9.6
    assert r.assumptions["matterpak_interior_footprint_sq_ft"] == 8800.0
    assert any("inside scan, not the roof" in c for c in r.caveats)


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
