"""Tests for backend/sun.py: sun geometry against known values, climate parsing, cache-only."""

from datetime import date, datetime, timezone

import pytest

from backend import config, sun


def test_noon_height_matches_textbook_rule() -> None:
    # Noon sun height = 90 - |latitude - declination|; Houston ~29.76 N.
    s = sun.sun_summary(29.76, -95.37)
    assert s["summer_solstice"]["noon_elevation_deg"] == pytest.approx(83.7, abs=0.2)
    assert s["winter_solstice"]["noon_elevation_deg"] == pytest.approx(36.8, abs=0.2)
    assert s["equinox"]["noon_elevation_deg"] == pytest.approx(60.2, abs=0.5)


def test_day_length_and_sunrise_direction() -> None:
    s = sun.sun_summary(29.76, -95.37)
    assert s["summer_solstice"]["day_length_hours"] == pytest.approx(14.07, abs=0.1)
    assert s["winter_solstice"]["day_length_hours"] == pytest.approx(10.23, abs=0.1)
    assert s["summer_solstice"]["sunrise_azimuth_deg"] < 90 < s["winter_solstice"]["sunrise_azimuth_deg"]


def test_sun_position_at_solar_noon_is_high_and_south() -> None:
    pos = sun.sun_position(29.76, -95.37, datetime(2026, 12, 21, 18, 19, tzinfo=timezone.utc))
    assert pos["elevation_deg"] == pytest.approx(36.8, abs=0.5)
    assert pos["azimuth_deg"] == pytest.approx(180, abs=3)


def test_sun_position_needs_timezone() -> None:
    with pytest.raises(ValueError):
        sun.sun_position(29.76, -95.37, datetime(2026, 6, 21, 12))


def test_sun_path_only_daylight() -> None:
    path = sun.sun_path(29.76, -95.37, date(2026, 6, 21))
    assert 13 <= len(path) <= 15 and all(p["elevation_deg"] > 0 for p in path)


def _power_payload() -> dict:
    def series(ann: float) -> dict:
        return {**{m: ann for m in sun.MONTHS}, "JUN": ann + 1, "DEC": ann - 1, "ANN": ann}
    return {"properties": {"parameter": {
        "ALLSKY_SFC_SW_DWN": series(5.0), "CLRSKY_SFC_SW_DWN": series(6.25),
        "CLOUD_AMT": series(40.0), "ALLSKY_KT": series(0.55), "T2M": {**series(20.0), "JUL": -999},
    }}}


class _Resp:
    def __init__(self, data: dict) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.data


def test_climate_fetches_once_then_uses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    calls = []
    monkeypatch.setattr(sun.requests, "get", lambda *a, **k: calls.append(k) or _Resp(_power_payload()))
    c = sun.climate(29.76, -95.37)
    assert c["sunlight_kwh_m2_day"] == 5.0 and c["sunlight_kwh_m2_year"] == 1825
    assert c["cloud_loss_percent"] == 20.0  # 1 - 5 / 6.25
    assert c["sunniest_month"] == "JUN" and c["darkest_month"] == "DEC"
    assert c["monthly"][6]["temperature_c"] is None  # -999 -> missing
    sun.climate(29.80, -95.40)  # same grid cell -> cache
    assert len(calls) == 1


def test_climate_cache_only_miss_is_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    monkeypatch.setattr(sun.requests, "get", lambda *a, **k: pytest.fail("network in cache-only"))
    assert sun.climate(29.76, -95.37) is None
