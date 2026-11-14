"""Solar estimate tests; PVWatts is never called."""

import pytest

from backend.schemas import Building, Measurement, SolarResult
from backend.tools import solar


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(solar.config, "NREL_API_KEY", None)
    monkeypatch.setattr(solar.config, "CACHE_DIR", tmp_path)


def test_fallback_payback_math_for_houston() -> None:
    b = Building(id=1, market_id=14, latitude=29.7, longitude=-95.4, name="Test Tower")
    m = Measurement(id="m-1", property_id=1, roof_area_sq_ft=10_000, observations={"rooftop_obstructions": "few"})
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


def test_pvwatts_result_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solar, "pvwatts_annual_kwh", lambda lat, lon, kw: 100_000.0)
    r = solar.estimate_solar(Building(id=2, market_id=20, latitude=34, longitude=-118),
                             Measurement(id="m-2", roof_area_sq_ft=10_000))
    assert r.annual_energy_kwh == 100_000
    assert r.assumptions["energy_source"] == "NREL PVWatts v8"


def test_missing_roof_gives_no_estimate() -> None:
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
