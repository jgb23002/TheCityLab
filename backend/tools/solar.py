"""Rooftop solar estimate, simple payback and ranking for retrofit mode.

Yearly output comes from NREL PVWatts v8 when NREL_API_KEY is set and the
network is allowed; otherwise a documented kWh-per-kW fallback is used. Every
number is an estimate and every assumption is returned with the result.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import requests

from backend import config, sun
from backend.schemas import Building, Measurement, SolarResult
from backend.tools import geometry

log = logging.getLogger(__name__)

PVWATTS_URL = "https://developer.nlr.gov/api/pvwatts/v8.json"
HOUSTON, LOS_ANGELES = 14, 20

# Share of measured roof that can hold panels after setbacks, HVAC and fire lanes.
USABLE_FRACTION = {"none": 0.70, "few": 0.60, "many": 0.45, None: 0.60}
WATTS_PER_USABLE_SQ_FT = 12.0  # flat-roof tilted racking with row spacing
INSTALLED_COST_PER_WATT = 2.00  # USD, commercial rooftop, before incentives
INCENTIVE_PERCENT = 0.0  # federal ITC eligibility depends on dates; not assumed
PANEL_TILT_DEG, PANEL_AZIMUTH_DEG, SYSTEM_LOSSES_PCT = 10, 180, 14
# Commercial electricity price, USD/kWh (approximate recent EIA state averages).
PRICE_PER_KWH = {HOUSTON: 0.09, LOS_ANGELES: 0.22}
DEFAULT_PRICE_PER_KWH = 0.13
# Fallback yearly kWh per installed kW when PVWatts is unavailable.
FALLBACK_KWH_PER_KW = {HOUSTON: 1300.0, LOS_ANGELES: 1550.0}
DEFAULT_KWH_PER_KW = 1300.0


def _cache_path(params: dict[str, Any]):
    """Return the cache file for one PVWatts request (the key is never stored)."""
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    return config.CACHE_DIR / "nrel" / f"pvwatts_{digest}.json"


def pvwatts_result(
    lat: float, lon: float, system_kw: float, azimuth_deg: float = PANEL_AZIMUTH_DEG,
) -> dict[str, Any] | None:
    """Return the cached/fetched PVWatts result (annual + monthly kWh, station info), or None."""
    params = {
        "system_capacity": round(system_kw, 2), "module_type": 0, "losses": SYSTEM_LOSSES_PCT,
        "array_type": 1, "tilt": PANEL_TILT_DEG, "azimuth": round(azimuth_deg, 1),
        "lat": round(lat, 5), "lon": round(lon, 5),
    }
    path = _cache_path(params)
    if path.exists():
        return json.loads(path.read_text())
    if config.USE_CACHE_ONLY or not config.NREL_API_KEY:
        return None
    try:
        resp = requests.get(PVWATTS_URL, params={**params, "api_key": config.NREL_API_KEY}, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        outputs = body["outputs"]
        result = {
            "params": params,
            "ac_annual": float(outputs["ac_annual"]),
            "ac_monthly": [float(v) for v in outputs.get("ac_monthly", [])] or None,
            "station_info": body.get("station_info"),
        }
    except Exception as exc:  # never log the URL: it contains the key
        log.warning("PVWatts failed (%s); using fallback yield", type(exc).__name__)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def pvwatts_annual_kwh(lat: float, lon: float, system_kw: float) -> float | None:
    """Return PVWatts yearly AC kWh for a roof system, or None if unavailable."""
    result = pvwatts_result(lat, lon, system_kw)
    return result["ac_annual"] if result else None


def payback(
    lat: float | None, lon: float | None, roof_area_sq_ft: float, market_id: int | None = None,
    obstructions: str | None = None, bearing_deg: float | None = None,
) -> dict[str, Any]:
    """Size a rooftop system for one roof and return output, cost, savings and payback.

    ``bearing_deg`` is the roof's azimuth (e.g. from
    ``backend.tools.vexcel.parcel_bearing_deg``); when unavailable, the stated
    due-south assumption (``PANEL_AZIMUTH_DEG``) is used instead and noted.
    """
    usable_fraction = USABLE_FRACTION.get(obstructions, USABLE_FRACTION[None])
    usable = roof_area_sq_ft * usable_fraction
    system_kw = usable * WATTS_PER_USABLE_SQ_FT / 1000
    azimuth = bearing_deg if bearing_deg is not None else PANEL_AZIMUTH_DEG
    pvwatts = (pvwatts_result(lat, lon, system_kw, azimuth)
               if lat is not None and lon is not None and system_kw >= 0.05 else None)
    source = "NREL PVWatts v8"
    if pvwatts is None:
        kwh = system_kw * FALLBACK_KWH_PER_KW.get(market_id, DEFAULT_KWH_PER_KW)
        source = "fallback kWh per kW (PVWatts unavailable)"
    else:
        kwh = pvwatts["ac_annual"]
    price = PRICE_PER_KWH.get(market_id, DEFAULT_PRICE_PER_KWH)
    cost = system_kw * 1000 * INSTALLED_COST_PER_WATT * (1 - INCENTIVE_PERCENT / 100)
    savings = kwh * price
    assumptions = {
        "usable_fraction_of_roof": usable_fraction,
        "watts_per_usable_sq_ft": WATTS_PER_USABLE_SQ_FT,
        "installed_cost_usd_per_watt": INSTALLED_COST_PER_WATT,
        "incentive_percent": INCENTIVE_PERCENT,
        "electricity_price_usd_per_kwh": price,
        "tilt_deg": PANEL_TILT_DEG, "azimuth_deg": azimuth,
        "azimuth_source": "Vexcel parcel polygon" if bearing_deg is not None else "assumed due south (no parcel polygon available)",
        "system_losses_pct": SYSTEM_LOSSES_PCT,
        "energy_source": source,
    }
    if pvwatts:
        assumptions["ac_monthly_kwh"] = pvwatts.get("ac_monthly")
        assumptions["weather_station"] = pvwatts.get("station_info")
    return {
        "usable_roof_area_sq_ft": round(usable),
        "capacity_kw": round(system_kw, 1),
        "annual_energy_kwh": round(kwh),
        "estimated_cost_usd": round(cost),
        "estimated_annual_savings_usd": round(savings),
        "estimated_payback_years": round(cost / savings, 1) if savings > 0 else None,
        "assumptions": assumptions,
    }


# observations["source"] values that represent a real measurement (Vexcel
# structure-footprint detection, parcel attributes) rather than a vision
# model's photo reading. Keep in sync with scripts/measure_sample.py's SOURCE.
_MEASURED_ROOF_SOURCES = ("Vexcel precomputed parcel attributes",)


def sun_context(lat: float | None, lon: float | None, tilt_deg: float, facing_deg: float) -> dict[str, Any] | None:
    """Sun geometry, NASA POWER climate and our own yearly kWh/m2 on this roof's tilt and facing.

    Pure math and public data (``backend.sun``), no model. When the climate is
    unknown (NASA POWER uncached in cache-only mode) the plane figure is skipped
    rather than reported from clear-sky assumptions.
    """
    if lat is None or lon is None:
        return None
    try:
        conditions = sun.solar_conditions(lat, lon)
    except Exception as exc:
        log.warning("sun context unavailable (%s)", type(exc).__name__)
        return None
    sky = conditions.get("sun") or {}
    climate = conditions.get("climate")
    facts: dict[str, Any] = {
        "best_fixed_tilt_deg": sky.get("best_fixed_tilt_deg"),
        "best_panel_azimuth_deg": sky.get("best_panel_azimuth_deg"),
        "summer_solstice": sky.get("summer_solstice"),
        "winter_solstice": sky.get("winter_solstice"),
        "method": sky.get("method"),
    }
    if climate:
        facts.update({
            "sunlight_kwh_m2_day": climate.get("sunlight_kwh_m2_day"),
            "clear_sky_kwh_m2_day": climate.get("clear_sky_kwh_m2_day"),
            "cloud_cover_percent": climate.get("cloud_cover_percent"),
            "cloud_loss_percent": climate.get("cloud_loss_percent"),
            "climate_source": climate.get("source"),
        })
        try:
            plane = sun.plane_sunlight(sun.sky_year(lat, lon), tilt_deg, facing_deg)
            facts.update({"plane_kwh_m2_year": plane.get("kwh_m2_year"),
                          "plane_tilt_deg": tilt_deg, "plane_facing_deg": facing_deg})
        except Exception as exc:
            log.warning("plane sunlight unavailable (%s)", type(exc).__name__)
    return facts


def estimate_solar(building: Building, vision_result: Measurement | None = None) -> SolarResult:
    """Estimate rooftop solar potential with explicit units and assumptions.

    Roof area prefers a real measurement when one is available: Vexcel's
    detected structure footprint on ``vision_result`` (see
    ``scripts/measure_sample.py``), else CoStar footprint math
    (``backend.tools.geometry``). Per the project's "math first, model last"
    rule, a vision model's photo-read roof area is never used as the roof-area
    source; ``vision_result`` otherwise only supplies caveats a formula can't
    derive (existing rooftop obstructions or panels visible in an aerial image).
    """
    sid = f"solar-{building.id}"
    obs = (vision_result.observations or {}) if vision_result else {}
    measured_roof = vision_result.roof_area_sq_ft if vision_result else None
    is_measured = obs.get("source") in _MEASURED_ROOF_SOURCES
    roof_source = "Vexcel detected structure footprint"
    if measured_roof and measured_roof > 0 and is_measured:
        roof = measured_roof
    else:
        roof = geometry.roof_footprint_sq_ft(building)
        roof_source = "CoStar building footprint fields (typical floor size, floor areas or gross building area divided by stories)"
    if not roof or roof <= 0:
        return SolarResult(id=sid, property_id=building.id,
                           caveats=["No roof footprint could be computed for this building, so no solar estimate was made."])
    obstructions = obs.get("rooftop_obstructions")
    bearing = obs.get("parcel_bearing_deg")
    est = payback(building.latitude, building.longitude, roof, building.market_id,
                  obstructions if isinstance(obstructions, str) else None,
                  bearing if isinstance(bearing, (int, float)) else None)
    caveats = [
        "All figures are estimates for screening, not a quote.",
        f"Roof area is estimated from {roof_source}, not a measured roof photo.",
        "Payback is simple payback: no financing, degradation, demand charges, O&M or tax incentives.",
    ]
    if obs.get("solar_panels_present") is True:
        caveats.append("The aerial image suggests solar panels are already on this roof.")
    if est["assumptions"]["energy_source"].startswith("fallback"):
        caveats.append("Yearly output uses a regional average because PVWatts was unavailable.")
    # Reported context only; height is not a payback input. Prefers a Matterport
    # room-scan median (real per-room measurement, see backend.tools.matterport)
    # when the caller supplied one, else CoStar story-count math.
    matterport_height = obs.get("matterport_building_height_ft")
    if isinstance(matterport_height, (int, float)) and matterport_height > 0:
        height, height_source = matterport_height, "Matterport room scan (median room height)"
    else:
        height = geometry.building_height_ft(building)
        height_source = "CoStar stories above grade times an assumed floor-to-floor height" if height else None
    # Interior cross-checks from the MatterPak point cloud, when the showcase batch fetched one.
    cross_checks = {key: obs[key] for key in ("matterpak_interior_height_ft", "matterpak_interior_footprint_sq_ft")
                    if isinstance(obs.get(key), (int, float)) and obs[key] > 0}
    if cross_checks:
        caveats.append("The Matterport point cloud gives an interior height and footprint cross-check; "
                       "it is an inside scan, not the roof.")
    # What the numbers mean to the owner: cost against the last sale, savings per unit per month.
    owner: dict[str, Any] = {}
    if building.last_sale_price and building.last_sale_price > 0:
        owner.update({"last_sale_price_usd": building.last_sale_price, "last_sale_date": building.last_sale_date,
                      "system_cost_pct_of_last_sale_price": round(100 * est["estimated_cost_usd"] / building.last_sale_price, 1)})
    if building.total_num_units and building.total_num_units > 0:
        owner["savings_per_unit_per_month_usd"] = round(est["estimated_annual_savings_usd"] / building.total_num_units / 12, 2)
    assumptions = {**est["assumptions"],
                   "estimated_cost_usd": est["estimated_cost_usd"],
                   "estimated_annual_savings_usd": est["estimated_annual_savings_usd"],
                   "estimated_payback_years": est["estimated_payback_years"],
                   "estimated_roof_area_sq_ft": roof, "roof_area_source": roof_source,
                   "estimated_building_height_ft": height, "building_height_source": height_source,
                   **cross_checks, **owner}
    sun_facts = sun_context(building.latitude, building.longitude, PANEL_TILT_DEG, est["assumptions"]["azimuth_deg"])
    if sun_facts:
        assumptions["sun"] = sun_facts
    return SolarResult(
        id=sid, property_id=building.id,
        usable_roof_area_sq_ft=est["usable_roof_area_sq_ft"], capacity_kw=est["capacity_kw"],
        annual_energy_kwh=est["annual_energy_kwh"], assumptions=assumptions,
        payback_years=est["estimated_payback_years"],
        score=est["estimated_payback_years"], caveats=caveats,
    )


def rank_buildings(estimates: list[SolarResult]) -> list[SolarResult]:
    """Rank by payback_years, falling back to legacy score; ties by bigger system.

    Buildings without a payback go last with rank None.
    """
    def years(estimate: SolarResult) -> float | None:
        """Prefer the shared payback field while accepting pre-integration estimates."""
        return estimate.payback_years if estimate.payback_years is not None else estimate.score

    scored = [e for e in estimates if years(e) is not None]
    unscored = [e for e in estimates if years(e) is None]
    scored.sort(key=lambda e: (years(e), -(e.annual_energy_kwh or 0)))
    ranked = [e.model_copy(update={"rank": i}) for i, e in enumerate(scored, start=1)]
    return ranked + [e.model_copy(update={"rank": None}) for e in unscored]
