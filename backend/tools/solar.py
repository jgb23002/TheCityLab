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

from backend import config
from backend.schemas import Building, Measurement, SolarResult

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


def pvwatts_annual_kwh(lat: float, lon: float, system_kw: float) -> float | None:
    """Return PVWatts yearly AC kWh for a roof system, or None if unavailable."""
    params = {
        "system_capacity": round(system_kw, 2), "module_type": 0, "losses": SYSTEM_LOSSES_PCT,
        "array_type": 1, "tilt": PANEL_TILT_DEG, "azimuth": PANEL_AZIMUTH_DEG,
        "lat": round(lat, 5), "lon": round(lon, 5),
    }
    path = _cache_path(params)
    if path.exists():
        return json.loads(path.read_text())["ac_annual"]
    if config.USE_CACHE_ONLY or not config.NREL_API_KEY:
        return None
    try:
        resp = requests.get(PVWATTS_URL, params={**params, "api_key": config.NREL_API_KEY}, timeout=20)
        resp.raise_for_status()
        ac_annual = float(resp.json()["outputs"]["ac_annual"])
    except Exception as exc:  # never log the URL: it contains the key
        log.warning("PVWatts failed (%s); using fallback yield", type(exc).__name__)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "ac_annual": ac_annual}))
    return ac_annual


def payback(
    lat: float | None, lon: float | None, roof_area_sq_ft: float, market_id: int | None = None,
    obstructions: str | None = None,
) -> dict[str, Any]:
    """Size a rooftop system for one roof and return output, cost, savings and payback."""
    usable_fraction = USABLE_FRACTION.get(obstructions, USABLE_FRACTION[None])
    usable = roof_area_sq_ft * usable_fraction
    system_kw = usable * WATTS_PER_USABLE_SQ_FT / 1000
    kwh = pvwatts_annual_kwh(lat, lon, system_kw) if lat is not None and lon is not None and system_kw >= 0.05 else None
    source = "NREL PVWatts v8"
    if kwh is None:
        kwh = system_kw * FALLBACK_KWH_PER_KW.get(market_id, DEFAULT_KWH_PER_KW)
        source = "fallback kWh per kW (PVWatts unavailable)"
    price = PRICE_PER_KWH.get(market_id, DEFAULT_PRICE_PER_KWH)
    cost = system_kw * 1000 * INSTALLED_COST_PER_WATT * (1 - INCENTIVE_PERCENT / 100)
    savings = kwh * price
    return {
        "usable_roof_area_sq_ft": round(usable),
        "capacity_kw": round(system_kw, 1),
        "annual_energy_kwh": round(kwh),
        "estimated_cost_usd": round(cost),
        "estimated_annual_savings_usd": round(savings),
        "estimated_payback_years": round(cost / savings, 1) if savings > 0 else None,
        "assumptions": {
            "usable_fraction_of_roof": usable_fraction,
            "watts_per_usable_sq_ft": WATTS_PER_USABLE_SQ_FT,
            "installed_cost_usd_per_watt": INSTALLED_COST_PER_WATT,
            "incentive_percent": INCENTIVE_PERCENT,
            "electricity_price_usd_per_kwh": price,
            "tilt_deg": PANEL_TILT_DEG, "azimuth_deg": PANEL_AZIMUTH_DEG,
            "system_losses_pct": SYSTEM_LOSSES_PCT,
            "energy_source": source,
        },
    }


def estimate_solar(building: Building, vision_result: Measurement) -> SolarResult:
    """Estimate rooftop solar potential with explicit units and assumptions.

    Uses the image-measured roof area only (never building floor area). The
    Payback is exposed in ``payback_years`` for the planner. The existing
    assumptions and score fields retain their payback/cost/savings values.
    """
    sid = f"solar-{building.id}"
    obs = vision_result.observations or {}
    roof = vision_result.roof_area_sq_ft
    if not roof or roof <= 0:
        return SolarResult(id=sid, property_id=building.id,
                           caveats=["No measured roof area for this building, so no solar estimate was made."])
    obstructions = obs.get("rooftop_obstructions")
    est = payback(building.latitude, building.longitude, roof, building.market_id,
                  obstructions if isinstance(obstructions, str) else None)
    caveats = [
        "All figures are estimates for screening, not a quote.",
        "Roof area is measured from an aerial image and may include roof areas that cannot hold panels.",
        "Payback is simple payback: no financing, degradation, demand charges, O&M or tax incentives.",
    ]
    if obs.get("solar_panels_present") is True:
        caveats.append("The aerial image suggests solar panels are already on this roof.")
    if est["assumptions"]["energy_source"].startswith("fallback"):
        caveats.append("Yearly output uses a regional average because PVWatts was unavailable.")
    assumptions = {**est["assumptions"],
                   "estimated_cost_usd": est["estimated_cost_usd"],
                   "estimated_annual_savings_usd": est["estimated_annual_savings_usd"],
                   "estimated_payback_years": est["estimated_payback_years"],
                   "measured_roof_area_sq_ft": roof}
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
