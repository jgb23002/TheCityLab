"""Heat score and cool-roof estimates: find big, dark, unshaded roofs and price the fix.

Heat score = roof darkness x roof size x lack of trees, scaled 0-100 within a cohort.
Roof colour is the vision model's judgement from photo colour, not measured reflectivity.
Cool-roof savings use a simple kWh-per-sq-ft rule (assumption listed); this is a screening
estimate, not an energy model, and gives no temperatures.
"""

from __future__ import annotations

import math
from typing import Any

from backend.schemas import Building, Measurement
from backend.tools import solar

DARKNESS = {"dark": 1.0, "medium": 0.6, "light": 0.3, "white": 0.1}
# Yearly cooling kWh saved per sq ft of roof when coated white, by starting colour.
# ~0.3 kWh/sq ft/yr dark->white for low-rise buildings in hot climates (LBNL-style range).
COOLING_KWH_SAVED_PER_SQ_FT = {"dark": 0.30, "medium": 0.15, "light": 0.05, "white": 0.0}
COATING_COST_PER_SQ_FT = (0.75, 1.50)  # EPA range, materials + labour
COATING_COST_MID = sum(COATING_COST_PER_SQ_FT) / 2
# Extra cost of a cool coating over an ordinary re-coat when the roof is due anyway.
COOL_PREMIUM_PER_SQ_FT = 0.20
SQ_FT_PER_M2 = 10.7639
PITCH_FACTS = [
    "EPA: more than 90% of US roofs are dark, low-reflectance surfaces.",
    "Monitored Sacramento buildings with reflective roofs used up to 40% less cooling energy.",
    "EPA: cool-roof coatings cost about $0.75-$1.50 per sq ft, materials and labour.",
]


def _roof_sq_ft(measurement: Measurement, facts: dict[str, Any] | None) -> tuple[float | None, str]:
    """Prefer Vexcel's measured footprint; fall back to the vision estimate."""
    if facts and facts.get("footprint_sq_ft"):
        return float(facts["footprint_sq_ft"]), "Vexcel footprint"
    if measurement.roof_area_sq_ft:
        return float(measurement.roof_area_sq_ft), "vision estimate"
    return None, "unknown"


def _tree_percent(measurement: Measurement, facts: dict[str, Any] | None) -> tuple[float | None, str]:
    """Prefer Vexcel parcel tree cover; fall back to the vision green-cover percent."""
    if facts and facts.get("parcel_tree_cover_percent") is not None:
        return float(facts["parcel_tree_cover_percent"]), "Vexcel parcel tree cover"
    if measurement.green_cover_percent is not None:
        return float(measurement.green_cover_percent), "vision green cover"
    return None, "unknown"


def heat_profile(building: Building, measurement: Measurement, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Raw heat inputs and an unscaled heat index for one building (None when roof colour or size is unknown)."""
    colour = (measurement.observations or {}).get("roof_color")
    roof, roof_src = _roof_sq_ft(measurement, facts)
    trees, tree_src = _tree_percent(measurement, facts)
    raw = None
    if colour in DARKNESS and roof:
        shade = 1 - min(max((trees or 0) / 100, 0), 1)
        raw = DARKNESS[colour] * math.sqrt(roof) * shade  # sqrt keeps one giant roof from swamping the list
    return {
        "property_id": building.id, "roof_color": colour, "roof_sq_ft": roof, "roof_source": roof_src,
        "roof_m2": round(roof / SQ_FT_PER_M2) if roof else None,
        "tree_cover_percent": trees, "tree_source": tree_src, "heat_index": raw,
    }


def cool_roof_estimate(building: Building, profile: dict[str, Any]) -> dict[str, Any]:
    """Coating cost, yearly cooling savings and simple payback for one roof."""
    colour, roof = profile.get("roof_color"), profile.get("roof_sq_ft")
    price = solar.PRICE_PER_KWH.get(building.market_id, solar.DEFAULT_PRICE_PER_KWH)
    if colour not in COOLING_KWH_SAVED_PER_SQ_FT or not roof:
        return {"cost_usd": None, "annual_kwh_saved": None, "annual_savings_usd": None, "payback_years": None,
                "note": "Roof colour or size unknown."}
    if colour == "white":
        return {"cost_usd": 0, "annual_kwh_saved": 0, "annual_savings_usd": 0, "payback_years": None,
                "note": "Roof already looks white; a coating adds little."}
    kwh = roof * COOLING_KWH_SAVED_PER_SQ_FT[colour]
    cost = roof * COATING_COST_MID
    savings = kwh * price
    return {
        "cost_usd": round(cost), "cost_range_usd": [round(roof * c) for c in COATING_COST_PER_SQ_FT],
        "annual_kwh_saved": round(kwh), "annual_savings_usd": round(savings),
        "payback_years": round(cost / savings, 1) if savings else None,
        "extra_cost_if_recoating_usd": round(roof * COOL_PREMIUM_PER_SQ_FT),
        "payback_if_recoating_years": round(roof * COOL_PREMIUM_PER_SQ_FT / savings, 1) if savings else None,
        "assumptions": {"cooling_kwh_saved_per_sq_ft": COOLING_KWH_SAVED_PER_SQ_FT[colour],
                        "coating_cost_usd_per_sq_ft": COATING_COST_MID,
                        "cool_premium_over_regular_recoat_usd_per_sq_ft": COOL_PREMIUM_PER_SQ_FT,
                        "electricity_price_usd_per_kwh": price},
    }


def rank_heat(
    buildings: list[Building], measurements: list[Measurement], facts_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Heat score 0-100 (hottest = 100) and cool-roof estimate per building, hottest first."""
    by_id = {m.property_id: m for m in measurements}
    rows = []
    for b in buildings:
        m = by_id.get(b.id)
        if m is None:
            continue
        p = heat_profile(b, m, (facts_by_id or {}).get(b.id))
        p["cool_roof"] = cool_roof_estimate(b, p)
        rows.append(p)
    top = max((r["heat_index"] for r in rows if r["heat_index"] is not None), default=0)
    for r in rows:
        r["heat_score"] = round(100 * r["heat_index"] / top) if top and r["heat_index"] is not None else None
    rows.sort(key=lambda r: -1 if r["heat_score"] is None else -r["heat_score"])
    for i, r in enumerate(rows, start=1):
        r["heat_rank"] = i if r["heat_score"] is not None else None
        r["summary"] = heat_sentence(r)
    return rows


def heat_sentence(row: dict[str, Any]) -> str:
    """Demo text, e.g. 'Dark roof, 4,200 m2, 3% tree cover. A cool-roof coating costs an estimated $X...'."""
    if row.get("heat_score") is None:
        return "Not enough roof information for a heat estimate."
    trees = "unknown" if row["tree_cover_percent"] is None else f"{row['tree_cover_percent']:g}%"
    text = f"{row['roof_color'].capitalize()} roof, {row['roof_m2']:,} m², {trees} tree cover."
    cr = row["cool_roof"]
    if cr.get("payback_years"):
        text += (f" A cool-roof coating costs an estimated ${cr['cost_usd']:,} (saves about ${cr['annual_savings_usd']:,} a year)."
                 f" If the roof is due for re-coating anyway, the extra ${cr['extra_cost_if_recoating_usd']:,} for a cool"
                 f" coating pays back in about {cr['payback_if_recoating_years']:g} years (estimated).")
    return text


def compare_retrofits(
    building: Building, measurement: Measurement, facts: dict[str, Any] | None = None,
    carport: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Roof solar vs solar carport vs cool roof, shortest payback first (None paybacks last)."""
    options = []
    roof, _ = _roof_sq_ft(measurement, facts)
    if roof:
        m = measurement.model_copy(update={"roof_area_sq_ft": roof})
        if facts:
            from backend.tools.roof_facts import obstruction_level
            level = obstruction_level(facts)
            if level:
                m = m.model_copy(update={"observations": {**(m.observations or {}), "rooftop_obstructions": level}})
        s = solar.estimate_solar(building, m)
        a = s.assumptions or {}
        options.append({"option": "Rooftop solar", "cost_usd": a.get("estimated_cost_usd"),
                        "annual_savings_usd": a.get("estimated_annual_savings_usd"),
                        "payback_years": a.get("estimated_payback_years"), "detail": f"{s.capacity_kw} kW on the roof"})
        if facts and facts.get("existing_solar"):
            options[-1]["detail"] += " (Vexcel sees solar already on this roof)"
    if carport and carport.get("payback_years") is not None:
        options.append({"option": "Solar carport", **{k: carport.get(k) for k in ("cost_usd", "annual_savings_usd", "payback_years")},
                        "detail": f"{carport.get('capacity_kw')} kW over {carport.get('spaces_covered')} spaces"})
    cr = cool_roof_estimate(building, heat_profile(building, measurement, facts))
    options.append({"option": "Cool roof coating", "cost_usd": cr.get("cost_usd"),
                    "annual_savings_usd": cr.get("annual_savings_usd"), "payback_years": cr.get("payback_years"),
                    "payback_if_recoating_years": cr.get("payback_if_recoating_years"),
                    "detail": cr.get("note") or "Reflective coating on the existing roof"})
    options.sort(key=lambda o: (o["payback_years"] is None, o["payback_years"] or 0))
    return options
