"""Retrofit upgrade screening for one building: estimate only what the inputs support.

Rooftop solar comes from the building's solar_payback estimate (backend/tools/solar.py).
A solar carport is estimated from the aerial parking-space count with the assumptions
below. LED lighting, window film and HVAC need inputs this pipeline does not have
(fixture inventory, glazing area, equipment data), so they are listed as not estimated
with the missing input named. Simple payback ignores incentives, financing, demand
charges and operating costs; it is a screening number, not a feasibility study.
"""

from __future__ import annotations

from typing import Any

from backend.tools import solar

# Solar carport assumptions (screening placeholders; replace with a local quote).
SQ_FT_PER_STALL = 162.0           # 9 ft x 18 ft standard stall
CARPORT_COVERAGE = 0.6            # share of stalls a canopy can realistically cover (aisles, trees, lighting)
CARPORT_WATTS_PER_SQ_FT = 15.0    # flat canopy modules, no inter-row spacing
CARPORT_COST_PER_WATT = 3.00      # USD; canopy steel and foundations cost more than rooftop racking

RANKING_BASIS = ("Estimated upgrades are ranked by simple payback (shortest first). Upgrades that could "
                 "not be estimated are listed separately with the missing input; they are not ranked.")
CAVEATS = [
    "Simple payback ignores incentives, tax credits, financing, demand charges and operating costs; "
    "it is a screening estimate, not a complete financial feasibility assessment.",
    "Inputs come from aerial imagery and CoStar records; an on-site assessment and utility bills are needed.",
]


def _assumption(value: Any, source: str) -> dict[str, Any]:
    """An assumption with its source, for display."""
    return {"value": value, "source": source}


def _rooftop(solar_result: dict[str, Any] | None) -> dict[str, Any]:
    """Rooftop solar from the building's solar_payback result."""
    if not solar_result or solar_result.get("skipped") or solar_result.get("capacity_kw") is None:
        reason = (solar_result or {}).get("reason") or "no rooftop solar estimate (needs an image-measured roof area)"
        return {"name": "Rooftop solar", "status": "not_estimated", "reason": reason}
    a = solar_result.get("assumptions") or {}
    payback = solar_result.get("payback_years")
    return {
        "name": "Rooftop solar", "status": "estimated",
        "capacity_kw": solar_result.get("capacity_kw"), "annual_kwh": solar_result.get("annual_energy_kwh"),
        "cost_usd": a.get("estimated_cost_usd"), "annual_savings_usd": a.get("estimated_annual_savings_usd"),
        "simple_payback_years": payback if payback is not None else a.get("estimated_payback_years"),
        "formula": "kW = usable roof sq ft x W per sq ft / 1000; savings = kWh x $/kWh; payback = cost / savings",
        "assumptions": {
            "usable_roof_area_sq_ft": _assumption(solar_result.get("usable_roof_area_sq_ft"), "image-measured roof x usable fraction"),
            "usable_fraction_of_roof": _assumption(a.get("usable_fraction_of_roof"), "backend/tools/solar.py (rooftop obstructions)"),
            "watts_per_usable_sq_ft": _assumption(a.get("watts_per_usable_sq_ft"), "backend/tools/solar.py"),
            "installed_cost_usd_per_watt": _assumption(a.get("installed_cost_usd_per_watt"), "backend/tools/solar.py"),
            "electricity_price_usd_per_kwh": _assumption(a.get("electricity_price_usd_per_kwh"), "backend/tools/solar.py (state average)"),
            "energy_source": _assumption(a.get("energy_source"), "NREL PVWatts v8 when configured, else regional kWh per kW"),
        },
    }


def _carport(spaces: int | None, spaces_source: str, market_id: int | None,
             kwh_per_kw: float | None, kwh_source: str) -> dict[str, Any]:
    """Solar carport over a share of the surface parking stalls."""
    if not spaces:
        return {"name": "Solar carport", "status": "not_estimated",
                "reason": "no known surface parking-space count for this building"}
    canopy_sq_ft = spaces * CARPORT_COVERAGE * SQ_FT_PER_STALL
    kw = canopy_sq_ft * CARPORT_WATTS_PER_SQ_FT / 1000
    yield_kwh_per_kw = kwh_per_kw or solar.FALLBACK_KWH_PER_KW.get(market_id, solar.DEFAULT_KWH_PER_KW)
    price = solar.PRICE_PER_KWH.get(market_id, solar.DEFAULT_PRICE_PER_KWH)
    kwh = kw * yield_kwh_per_kw
    cost = kw * 1000 * CARPORT_COST_PER_WATT
    savings = kwh * price
    return {
        "name": "Solar carport", "status": "estimated",
        "capacity_kw": round(kw, 1), "annual_kwh": round(kwh), "cost_usd": round(cost),
        "annual_savings_usd": round(savings), "simple_payback_years": round(cost / savings, 1) if savings > 0 else None,
        "formula": ("canopy sq ft = spaces x coverage x 162; kW = canopy sq ft x W per sq ft / 1000; "
                    "kWh = kW x kWh per kW; payback = cost / (kWh x $/kWh)"),
        "assumptions": {
            "parking_spaces": _assumption(spaces, spaces_source),
            "canopy_coverage_of_stalls": _assumption(CARPORT_COVERAGE, "screening assumption"),
            "sq_ft_per_stall": _assumption(SQ_FT_PER_STALL, "standard 9 x 18 ft stall"),
            "watts_per_canopy_sq_ft": _assumption(CARPORT_WATTS_PER_SQ_FT, "screening assumption"),
            "installed_cost_usd_per_watt": _assumption(CARPORT_COST_PER_WATT, "screening assumption (canopy structure)"),
            "kwh_per_kw": _assumption(round(yield_kwh_per_kw, 1), kwh_source),
            "electricity_price_usd_per_kwh": _assumption(price, "backend/tools/solar.py (state average)"),
        },
    }


def screen(building: dict[str, Any], measurement: dict[str, Any] | None, solar_result: dict[str, Any] | None,
           interior: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screen the five upgrades for one building; returns JSON-safe results."""
    m = measurement or {}
    obs = m.get("observations") or {}
    rooftop = _rooftop(solar_result)
    kwh_per_kw, kwh_source = None, "regional fallback kWh per kW (backend/tools/solar.py)"
    if rooftop["status"] == "estimated" and rooftop.get("capacity_kw") and rooftop.get("annual_kwh"):
        kwh_per_kw = rooftop["annual_kwh"] / rooftop["capacity_kw"]
        kwh_source = "same building's rooftop solar estimate (kWh per kW)"
    spaces, spaces_source = m.get("parking_space_count"), "aerial image measurement"
    if spaces is None and building.get("surface_parking_spaces"):
        spaces, spaces_source = building["surface_parking_spaces"], "CoStar surface parking record"
    carport = _carport(spaces, spaces_source, building.get("market_id"), kwh_per_kw, kwh_source)

    scan = interior if interior and interior.get("available") else None
    scanned = (f" The Matterport scan measured {scan['summary'].get('scanned_floor_area')} "
               f"{scan['summary'].get('units') or ''} of floor area but records no fixtures.") if scan else ""
    hvac_hint = obs.get("rooftop_obstructions")
    upgrades = [rooftop, carport,
                {"name": "LED lighting", "status": "not_estimated",
                 "reason": "needs a lighting fixture inventory, wattage and operating hours." + scanned},
                {"name": "Window film", "status": "not_estimated",
                 "reason": "needs glazing area, orientation and cooling load; top-down imagery does not show facades."},
                {"name": "HVAC upgrade", "status": "not_estimated",
                 "reason": "needs equipment type, age and efficiency data." + (
                     f" Imagery shows '{hvac_hint}' rooftop obstructions (which can include HVAC units), "
                     "an indicator only." if hvac_hint else "")}]
    estimated = sorted([u for u in upgrades if u["status"] == "estimated" and u.get("simple_payback_years") is not None],
                       key=lambda u: u["simple_payback_years"])
    for rank, u in enumerate(estimated, start=1):
        u["rank"] = rank
    return {"property_id": building.get("id"), "upgrades": upgrades, "ranking_basis": RANKING_BASIS,
            "estimated_count": len(estimated), "caveats": CAVEATS}
