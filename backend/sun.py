"""Sun position and local solar climate for a building's latitude/longitude.

Two parts, both math or public data, no model:

1. **Sun position** (``sun_position``, ``sun_summary``): the NOAA solar
   position equations (declination, equation of time, hour angle). Gives the
   sun's height and compass direction at any moment, solar noon height on the
   solstices and equinox, day length and sunrise/sunset directions.
2. **Climate** (``climate``): NASA POWER long-term monthly averages
   (https://power.larc.nasa.gov, free, no key): sunlight reaching the ground
   (all-sky), sunlight on a cloudless day (clear-sky), cloud cover %,
   clearness index and air temperature. ``cloud_loss_percent`` is how much
   sunlight clouds remove: 1 - all_sky / clear_sky.

NASA POWER cells are about 0.5 deg x 0.625 deg (~50 km), so buildings in one
city share values; it describes the city's weather, not a single roof.
Every response is cached under ``data/cache/nasa_power/``; in cache-only mode
a missing cell returns None.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from backend import config

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
POWER_PARAMS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,CLOUD_AMT,ALLSKY_KT,T2M"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
# Sun's centre 0.833 deg below the horizon at sunrise/sunset (refraction + radius).
HORIZON_DEG = -0.833


# ---------------------------------------------------------------- sun position

def _solar_terms(when: datetime) -> tuple[float, float]:
    """Declination (deg) and equation of time (minutes) for a UTC moment (NOAA)."""
    when = when.astimezone(timezone.utc)
    day = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    leap = 366 if (when.year % 4 == 0 and (when.year % 100 != 0 or when.year % 400 == 0)) else 365
    g = 2 * math.pi / leap * (day - 1 + (hour - 12) / 24)  # fractional year, radians
    eot = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    return math.degrees(decl), eot


def sun_position(lat: float, lon: float, when: datetime) -> dict[str, float]:
    """Sun height above the horizon and compass direction (0 = north, 180 = south).

    ``when`` must be timezone-aware. Accurate to about 0.5 deg, plenty for
    roof screening.
    """
    if when.tzinfo is None:
        raise ValueError("when must be timezone-aware")
    utc = when.astimezone(timezone.utc)
    decl, eot = _solar_terms(utc)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    true_solar_min = minutes + eot + 4 * lon
    hour_angle = math.radians(true_solar_min / 4 - 180)
    phi, d = math.radians(lat), math.radians(decl)
    cos_zen = math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(hour_angle)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.degrees(math.atan2(math.sin(hour_angle),
                                 math.cos(hour_angle) * math.sin(phi) - math.tan(d) * math.cos(phi)))
    return {"elevation_deg": round(90 - math.degrees(zen), 2),
            "azimuth_deg": round((az + 180) % 360, 2)}


def _day_facts(lat: float, lon: float, day: date) -> dict[str, Any]:
    """Noon height, day length, sunrise/sunset direction and solar noon (UTC) for one date."""
    noon_guess = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
    decl, eot = _solar_terms(noon_guess)
    phi, d = math.radians(lat), math.radians(decl)
    solar_noon_utc = noon_guess.replace(hour=0) + timedelta(minutes=720 - 4 * lon - eot)
    cos_h0 = ((math.sin(math.radians(HORIZON_DEG)) - math.sin(phi) * math.sin(d))
              / (math.cos(phi) * math.cos(d)))
    if cos_h0 <= -1:
        day_hours = 24.0
    elif cos_h0 >= 1:
        day_hours = 0.0
    else:
        day_hours = 2 * math.degrees(math.acos(cos_h0)) / 15
    cos_az = max(-1.0, min(1.0, math.sin(d) / math.cos(phi)))
    rise_az = math.degrees(math.acos(cos_az))
    return {
        "date": day.isoformat(),
        "declination_deg": round(decl, 2),
        "noon_elevation_deg": round(90 - abs(lat - decl), 1),
        "day_length_hours": round(day_hours, 2),
        "sunrise_azimuth_deg": round(rise_az, 1),
        "sunset_azimuth_deg": round(360 - rise_az, 1),
        "solar_noon_utc": solar_noon_utc.strftime("%H:%M"),
    }


def sun_path(lat: float, lon: float, day: date, step_minutes: int = 60) -> list[dict[str, Any]]:
    """Sun position through one day (UTC times), only while the sun is up."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    points = []
    for i in range(0, 24 * 60, step_minutes):
        when = start + timedelta(minutes=i)
        pos = sun_position(lat, lon, when)
        if pos["elevation_deg"] > 0:
            points.append({"utc": when.strftime("%H:%M"), **pos})
    return points


def sun_summary(lat: float, lon: float, year: int = 2026) -> dict[str, Any]:
    """Sun geometry for screening a flat roof at this location.

    ``best_fixed_tilt_deg`` uses the common rule tilt ~ 0.76 x latitude + 3.1
    (Jacobson & Jadhav 2018 fit for fixed panels); north of the tropics the
    best direction is due south (180 deg).
    """
    return {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "summer_solstice": _day_facts(lat, lon, date(year, 6, 21)),
        "equinox": _day_facts(lat, lon, date(year, 3, 20)),
        "winter_solstice": _day_facts(lat, lon, date(year, 12, 21)),
        "best_panel_azimuth_deg": 180.0 if lat >= 0 else 0.0,
        "best_fixed_tilt_deg": round(0.76 * abs(lat) + 3.1, 1),
        "method": "NOAA solar position equations (no model)",
    }


# --------------------------------------------------------------------- climate

def _cell(lat: float, lon: float) -> tuple[float, float]:
    """Snap to NASA POWER's ~0.5 deg grid so nearby buildings share one request."""
    return round(round(lat * 2) / 2, 1), round(round(lon * 1.6) / 1.6, 3)


def _power_cache_path(cell: tuple[float, float]):
    return config.CACHE_DIR / "nasa_power" / f"climatology_{cell[0]}_{cell[1]}.json"


def _fetch_power(lat: float, lon: float) -> dict[str, Any] | None:
    """NASA POWER climatology for the grid cell around (lat, lon), cache first."""
    cell = _cell(lat, lon)
    path = _power_cache_path(cell)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if config.USE_CACHE_ONLY:
        return None
    params = {"parameters": POWER_PARAMS, "community": "RE", "latitude": cell[0],
              "longitude": cell[1], "format": "JSON"}
    try:
        resp = requests.get(POWER_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        data["properties"]["parameter"]  # shape check before caching
    except Exception as exc:
        log.warning("NASA POWER failed (%s)", type(exc).__name__)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _value(series: dict[str, Any], key: str) -> float | None:
    v = series.get(key)
    return float(v) if isinstance(v, (int, float)) and v > -900 else None  # -999 = missing


def climate(lat: float, lon: float) -> dict[str, Any] | None:
    """Long-term sunlight, cloud cover and temperature for a location, or None."""
    data = _fetch_power(lat, lon)
    if data is None:
        return None
    p = data["properties"]["parameter"]

    def pick(name: str, key: str) -> float | None:
        return _value(p.get(name) or {}, key)

    def cloud_loss(key: str) -> float | None:
        all_sky, clear = pick("ALLSKY_SFC_SW_DWN", key), pick("CLRSKY_SFC_SW_DWN", key)
        return round(100 * (1 - all_sky / clear), 1) if all_sky is not None and clear else None

    monthly = [{
        "month": m,
        "sunlight_kwh_m2_day": pick("ALLSKY_SFC_SW_DWN", m),
        "clear_sky_kwh_m2_day": pick("CLRSKY_SFC_SW_DWN", m),
        "cloud_cover_percent": pick("CLOUD_AMT", m),
        "cloud_loss_percent": cloud_loss(m),
        "temperature_c": pick("T2M", m),
    } for m in MONTHS]
    sunlight = pick("ALLSKY_SFC_SW_DWN", "ANN")
    sunny = [r for r in monthly if r["sunlight_kwh_m2_day"] is not None]
    return {
        "grid_cell": list(_cell(lat, lon)),
        "sunlight_kwh_m2_day": sunlight,
        "sunlight_kwh_m2_year": round(sunlight * 365) if sunlight is not None else None,
        "clear_sky_kwh_m2_day": pick("CLRSKY_SFC_SW_DWN", "ANN"),
        "cloud_cover_percent": pick("CLOUD_AMT", "ANN"),
        "cloud_loss_percent": cloud_loss("ANN"),
        "clearness_index": pick("ALLSKY_KT", "ANN"),
        "temperature_c": pick("T2M", "ANN"),
        "sunniest_month": max(sunny, key=lambda r: r["sunlight_kwh_m2_day"])["month"] if sunny else None,
        "darkest_month": min(sunny, key=lambda r: r["sunlight_kwh_m2_day"])["month"] if sunny else None,
        "monthly": monthly,
        "source": "NASA POWER climatology (long-term monthly averages), ~50 km grid cell",
    }


def solar_conditions(lat: float, lon: float) -> dict[str, Any]:
    """Sun geometry plus local climate for one location (climate may be None)."""
    return {"sun": sun_summary(lat, lon), "climate": climate(lat, lon)}
