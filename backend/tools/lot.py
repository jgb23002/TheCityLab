"""Lot fullness ("ghost building" flag) and the Empty Lot Planner.

Lot fullness = cars seen / parking spaces, from one aerial photo. Compared with CoStar's
reported vacancy it flags "leased but looks empty" buildings as worth a closer look,
never as failing. The planner then sizes the spare parking and ranks what an owner
could do with it, every assumption shown. The transit rule check is a "likely
eligible" flag, not a legal answer.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from datetime import date
from functools import lru_cache
from typing import Any

from backend import config, llm
from backend.schemas import Building, Measurement
from backend.tools import solar

log = logging.getLogger(__name__)

HOUSTON, LOS_ANGELES = 14, 20
BUFFER = 0.25  # keep 25% more spaces than cars seen
M2_PER_SPACE = 30.0  # stall plus share of drive aisle
HALF_MILE_M = 804.7
STATIONS_CSV = config.PROJECT_ROOT / "data" / "transit" / "rail_stations.csv"
# Approximate boxes (min_lat, min_lon, max_lat, max_lon) of Houston's market-based
# (no-minimum) parking areas: Downtown, East Downtown, parts of Midtown.
HOUSTON_NO_MINIMUM_AREAS = {
    "Downtown": (29.745, -95.378, 29.772, -95.350),
    "East Downtown": (29.735, -95.355, 29.760, -95.335),
    "Midtown (part)": (29.732, -95.392, 29.752, -95.372),
}
MONTHLY_RENT_PER_SPACE = {HOUSTON: 60.0, LOS_ANGELES: 120.0}
DEFAULT_MONTHLY_RENT = 75.0
RENTED_SHARE = 0.5  # half the spare spaces find a renter
PLATFORM_FEE = 0.2
EV_PORTS_MAX = 10
EV_HOST_REVENUE_PER_PORT = 300.0  # USD/year revenue share, operator-funded equipment
CARPORT_KW_PER_SPACE = 3.5
CARPORT_COST_PER_WATT = 3.00
CARPORT_SPACES_MAX = 60
GREEN_SHARE_OF_SPARE = 0.2
GREEN_COST_PER_SQ_FT = 8.0
HOUSTON_DRAINAGE_USD_PER_SQ_FT = 0.032  # yearly City of Houston drainage charge on impervious area
SQ_FT_PER_M2 = 10.7639
PAD_SITE_MIN_M2 = 465.0  # ~5,000 sq ft


# ---------- lot fullness ----------

def _weekday(day: str | None) -> tuple[str | None, bool | None]:
    """('Tue Mar 12 2025', True) for an ISO date; weekday flag is False on weekends."""
    if not day:
        return None, None
    try:
        d = date.fromisoformat(day[:10])
    except ValueError:
        return None, None
    return d.strftime("%a %b %-d %Y"), d.weekday() < 5


def lot_fullness(building: Building, measurement: Measurement, facts: dict[str, Any] | None = None,
                 capture_date: str | None = None) -> dict[str, Any]:
    """Cars / spaces for one property, compared with CoStar vacancy, with a flag and caveats."""
    obs = measurement.observations or {}
    cars, car_src = obs.get("cars_seen"), "vision (Vexcel boxes checked by Claude)"
    if cars is None and facts and facts.get("vehicles_count") is not None:
        cars, car_src = facts["vehicles_count"], "Vexcel vehicle count"
    seen = measurement.parking_space_count
    costar = building.surface_parking_spaces
    note = None
    if seen == 0:
        spaces, space_src, note = None, "none visible", "No surface lot was visible in the photo, so fullness is not judged."
    elif seen and seen >= max(cars or 0, 4):
        spaces, space_src = seen, "aerial count"
    elif costar and car_src == "Vexcel vehicle count" and costar >= (cars or 0):
        spaces, space_src = costar, "CoStar surface spaces"  # both whole-parcel counts
    else:
        spaces, space_src, note = None, "unreliable", ("The photo shows too few spaces to compare with the cars seen; "
                                                        "the lot may extend beyond the ~80 m image.")
    day = capture_date or (facts or {}).get("ortho_date")
    day_text, is_weekday = _weekday(day)
    fullness = min(100.0, round(100 * cars / spaces, 1)) if cars is not None and spaces else None
    occupancy = round(100 - building.vacancy_percent, 1) if building.vacancy_percent is not None else None

    flag = "unknown"
    if fullness is not None and occupancy is not None:
        if is_weekday is False:
            flag = "weekend photo, not judged"
        elif occupancy >= 80 and fullness <= 30:
            flag = "leased but looks empty"
        elif abs(occupancy - fullness) <= 40:
            flag = "matches"
        else:
            flag = "differs"
    caveats = ["One photo is one moment: worth a closer look, not proof a building is failing.",
               "Garages hide cars and shared lots can serve several buildings."]
    if is_weekday is False:
        caveats.append("Photo was taken on a weekend; office lots look emptier then.")
    if day and ((day[5:7] == "12" and day[8:10] >= "20") or (day[5:7] == "01" and day[8:10] <= "02")):
        caveats.append("Photo was taken in the holiday week; lots may look emptier than usual.")
    if note:
        caveats.insert(0, note)
    if building.property_type == 11:
        caveats.append("Apartment lots empty out on weekdays while residents are at work.")
    text = None
    if fullness is not None:
        lead = f"Reported: {occupancy:g}% leased. " if occupancy is not None else ""
        text = f"{lead}Lot{(' on ' + day_text) if day_text else ''}: {fullness:g}% full ({cars} of {spaces} spaces)."
    return {"property_id": building.id, "cars": cars, "cars_source": car_src, "spaces": spaces,
            "spaces_source": space_src, "lot_fullness_percent": fullness, "reported_occupancy_percent": occupancy,
            "capture_date": day[:10] if day else None, "capture_weekday": is_weekday, "flag": flag,
            "summary": text, "caveats": caveats}


def fullness_vs_vacancy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Across buildings: does lot fullness line up with reported occupancy? (Pearson and Spearman)."""
    pairs = [(r["lot_fullness_percent"], r["reported_occupancy_percent"]) for r in rows
             if r.get("lot_fullness_percent") is not None and r.get("reported_occupancy_percent") is not None
             and r.get("capture_weekday") is not False]
    out: dict[str, Any] = {"n": len(pairs), "flagged_leased_but_empty": sum(1 for r in rows if r.get("flag") == "leased but looks empty")}
    if len(pairs) < 5:
        out["interpretation"] = "Too few weekday buildings with both numbers to test."
        return out
    from scipy import stats as st

    x, y = zip(*pairs)
    pr, pp = st.pearsonr(x, y)
    sr, sp = st.spearmanr(x, y)
    out.update(pearson_r=round(float(pr), 3), pearson_p=round(float(pp), 4),
               spearman_rho=round(float(sr), 3), spearman_p=round(float(sp), 4))
    strength = "no clear" if pp >= 0.05 else ("a positive" if pr > 0 else "a negative")
    out["interpretation"] = (f"Across {len(pairs)} buildings, lot fullness shows {strength} link with reported "
                             f"occupancy (r = {pr:.2f}, p = {pp:.3f}). Observational, one photo per building.")
    return out


# ---------- transit / parking-minimum check ----------

@lru_cache(maxsize=1)
def _stations() -> list[dict[str, Any]]:
    """Rail stations bundled in data/transit/ (works offline)."""
    try:
        with STATIONS_CSV.open(newline="") as f:
            return [{**r, "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(f)]
    except OSError:
        log.warning("Transit station file missing: %s", STATIONS_CSV)
        return []


def _metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * math.asin(math.sqrt(a))


def parking_rule_check(building: Building) -> dict[str, Any]:
    """Can spaces likely be removed? LA: AB 2097 near rail. Houston: no-minimum areas."""
    lat, lon = building.latitude, building.longitude
    if lat is None or lon is None:
        return {"can_remove_spaces": False, "reason": "No location, so parking must be kept.", "nearest_station": None}
    city = {HOUSTON: "Houston", LOS_ANGELES: "Los Angeles"}.get(building.market_id)
    near = None
    for s in _stations():
        if city and s["city"] != city:
            continue
        d = _metres(lat, lon, s["lat"], s["lon"])
        if near is None or d < near[1]:
            near = (s["name"], d)
    station = {"name": near[0], "miles": round(near[1] / 1609.34, 2)} if near else None
    within = bool(near and near[1] <= HALF_MILE_M)
    if building.market_id == LOS_ANGELES and within:
        return {"can_remove_spaces": True, "nearest_station": station,
                "reason": f"{station['miles']} miles from {station['name']} rail station, so AB 2097 likely removes the parking minimum."}
    if building.market_id == HOUSTON:
        for area, (a, b, c, d) in HOUSTON_NO_MINIMUM_AREAS.items():
            if a <= lat <= c and b <= lon <= d:
                return {"can_remove_spaces": True, "nearest_station": station,
                        "reason": f"Inside Houston's {area} area, where parking minimums were removed (approximate boundary)."}
        if within:
            return {"can_remove_spaces": False, "nearest_station": station,
                    "reason": f"{station['miles']} miles from {station['name']}; Houston may allow a parking reduction, but minimums likely apply."}
    return {"can_remove_spaces": False, "nearest_station": station,
            "reason": "No transit exemption found, so options that keep the parking come first."}


# ---------- Empty Lot Planner ----------

def _carport(building: Building, spaces: int) -> dict[str, Any] | None:
    """Solar canopy over up to CARPORT_SPACES_MAX existing spaces."""
    covered = min(spaces, CARPORT_SPACES_MAX)
    if covered < 4:
        return None
    kw = covered * CARPORT_KW_PER_SPACE
    kwh = None
    if building.latitude is not None and building.longitude is not None:
        kwh = solar.pvwatts_annual_kwh(building.latitude, building.longitude, kw)
    source = "NREL PVWatts v8"
    if kwh is None:
        kwh, source = kw * solar.FALLBACK_KWH_PER_KW.get(building.market_id, solar.DEFAULT_KWH_PER_KW), "regional average"
    price = solar.PRICE_PER_KWH.get(building.market_id, solar.DEFAULT_PRICE_PER_KWH)
    cost, savings = kw * 1000 * CARPORT_COST_PER_WATT, kwh * price
    return {"spaces_covered": covered, "capacity_kw": round(kw, 1), "annual_kwh": round(kwh), "energy_source": source,
            "cost_usd": round(cost), "annual_savings_usd": round(savings), "payback_years": round(cost / savings, 1) if savings else None}


def plan_empty_lot(building: Building, fullness: dict[str, Any]) -> dict[str, Any]:
    """Spare spaces, spare land, rule check and ranked options for one lot."""
    spaces, cars = fullness.get("spaces"), fullness.get("cars")
    if not spaces or cars is None or fullness.get("lot_fullness_percent") is None:
        return {"property_id": building.id, "options": [], "recommendation": None,
                "caveats": ["Need both a space count and a car count to plan the lot."]}
    keep = min(spaces, math.ceil(cars * (1 + BUFFER)))
    spare = spaces - keep
    land_m2 = spare * M2_PER_SPACE
    rule = parking_rule_check(building)
    rent = MONTHLY_RENT_PER_SPACE.get(building.market_id, DEFAULT_MONTHLY_RENT)
    rented = round(spare * RENTED_SHARE)
    carport = _carport(building, spaces)
    green_sq_ft = land_m2 * GREEN_SHARE_OF_SPARE * SQ_FT_PER_M2
    drainage = green_sq_ft * HOUSTON_DRAINAGE_USD_PER_SQ_FT if building.market_id == HOUSTON else None

    options = [
        {"option": "Rent the spaces", "keeps_parking": "Yes", "speed": "Days", "cost_usd": 0,
         "annual_income_usd": round(rented * rent * 12 * (1 - PLATFORM_FEE)) if rented else 0,
         "detail": f"Rent {rented} of {spare} spare spaces at about ${rent:g}/month after a {PLATFORM_FEE:.0%} app fee."},
        {"option": "EV charging", "keeps_parking": "Yes", "speed": "Months", "cost_usd": 0,
         "annual_income_usd": round(min(spare, EV_PORTS_MAX) * EV_HOST_REVENUE_PER_PORT) if spare >= 2 else 0,
         "detail": f"{min(spare, EV_PORTS_MAX)} operator-funded chargers with a revenue share."},
    ]
    if carport:
        options.append({"option": "Solar carport", "keeps_parking": "Yes", "speed": "Months", "cost_usd": carport["cost_usd"],
                        "annual_income_usd": carport["annual_savings_usd"], "payback_years": carport["payback_years"],
                        "detail": f"{carport['capacity_kw']} kW canopy over {carport['spaces_covered']} spaces ({carport['energy_source']})."})
    options.append({"option": "Go partly green", "keeps_parking": "Partly", "speed": "Months",
                    "cost_usd": round(green_sq_ft * GREEN_COST_PER_SQ_FT),
                    "annual_income_usd": round(drainage) if drainage is not None else None,
                    "detail": f"Turn {green_sq_ft:,.0f} sq ft into trees and rain gardens"
                              + (" (lower Houston drainage charge)." if drainage is not None else ".")})
    options.append({"option": "Temporary uses", "keeps_parking": "Partly", "speed": "Weeks", "cost_usd": None,
                    "annual_income_usd": None, "detail": "Food trucks, markets and pop-up events on spare spaces."})
    if rule["can_remove_spaces"] and land_m2 >= PAD_SITE_MIN_M2:
        options.append({"option": "Build on it", "keeps_parking": "No", "speed": "Years", "cost_usd": None,
                        "annual_income_usd": None, "detail": f"About {land_m2:,.0f} m² could fit a pad site or housing."})
    for o in options:
        if "payback_years" not in o:
            o["payback_years"] = round(o["cost_usd"] / o["annual_income_usd"], 1) if o["cost_usd"] and o["annual_income_usd"] else (0 if o["cost_usd"] == 0 and o["annual_income_usd"] else None)
    allowed = [o for o in options if o["keeps_parking"] != "No" or rule["can_remove_spaces"]]
    allowed.sort(key=lambda o: (o["annual_income_usd"] is None, o["payback_years"] if o["payback_years"] is not None else 99, -(o["annual_income_usd"] or 0)))
    plan = {"property_id": building.id, "spaces": spaces, "cars": cars, "keep_spaces": keep, "spare_spaces": spare,
            "spare_land_m2": round(land_m2), "rule_check": rule, "carport": carport, "options": allowed,
            "assumptions": {"buffer_over_cars": BUFFER, "m2_per_space": M2_PER_SPACE, "monthly_rent_per_space_usd": rent,
                            "rented_share": RENTED_SHARE, "ev_host_revenue_per_port_usd": EV_HOST_REVENUE_PER_PORT,
                            "carport_kw_per_space": CARPORT_KW_PER_SPACE, "carport_cost_usd_per_watt": CARPORT_COST_PER_WATT,
                            "green_cost_usd_per_sq_ft": GREEN_COST_PER_SQ_FT,
                            "houston_drainage_usd_per_sq_ft_year": HOUSTON_DRAINAGE_USD_PER_SQ_FT},
            "caveats": ["Based on one photo; check more days before acting.", "Dollar figures are rough estimates.",
                        "The transit check is a 'likely eligible' flag, not a legal answer."]}
    plan["recommendation"] = recommend(building, fullness, plan)
    return plan


RECOMMEND_PROMPT = """You advise a commercial property owner about an under-used parking lot.
Facts (use only these; call every number an estimate):
{facts}
Write 2-3 sentences: what the photo showed, the best one or two options from the
ranked list (keep parking unless can_remove_spaces is true), and the parking-rule reason.
No names, phone numbers, emails or links. Under 90 words. Plain text only."""


def template_recommendation(fullness: dict[str, Any], plan: dict[str, Any]) -> str:
    """Fixed-wording recommendation used when the model is unavailable."""
    opts = [o for o in plan["options"] if o.get("annual_income_usd")][:2]
    when = f" on {fullness['capture_date']}" if fullness.get("capture_date") else ""
    text = (f"This lot was about {fullness['lot_fullness_percent']:g}% full{when}. Keep {plan['keep_spaces']} spaces; "
            f"{plan['spare_spaces']} look spare (estimated).")
    if opts:
        text += " Best options: " + "; ".join(f"{o['option'].lower()} (about ${o['annual_income_usd']:,} a year, estimated)" for o in opts) + "."
    return text + " " + plan["rule_check"]["reason"]


def recommend(building: Building, fullness: dict[str, Any], plan: dict[str, Any]) -> str:
    """One Bedrock-written recommendation; falls back to the template offline or on bad output."""
    if fullness.get("lot_fullness_percent") is None:
        return "Not enough parking data for a recommendation."
    if config.USE_CACHE_ONLY:
        return template_recommendation(fullness, plan)
    facts = {"lot_fullness_percent": fullness["lot_fullness_percent"], "photo_date": fullness.get("capture_date"),
             "spaces": plan["spaces"], "cars": plan["cars"], "keep_spaces": plan["keep_spaces"],
             "spare_spaces": plan["spare_spaces"], "can_remove_spaces": plan["rule_check"]["can_remove_spaces"],
             "parking_rule": plan["rule_check"]["reason"],
             "ranked_options": [{k: o.get(k) for k in ("option", "annual_income_usd", "cost_usd", "payback_years", "detail")}
                                for o in plan["options"][:4]]}
    try:
        text = llm.ask_text(RECOMMEND_PROMPT.format(facts=json.dumps(facts, indent=1))).strip()
    except Exception as exc:
        log.warning("Recommendation model call failed: %s", type(exc).__name__)
        return template_recommendation(fullness, plan)
    if not text or len(text.split()) > 110 or re.search(r"@|https?://|\d{3}[\s.-]\d{3}[\s.-]\d{4}", text):
        return template_recommendation(fullness, plan)
    return text
