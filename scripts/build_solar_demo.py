"""Build the data file for the solar demo page (frontend/solar/): every apartment roof in a
city sample, what solar on it would save, and who owns it.

Run ``python scripts/build_solar_demo.py --city houston`` (add ``los_angeles``).

Per building, math only (no model):

1. Building outlines and roof shapes from Vexcel (the matched parcel; cached).
2. Each pitched roof split into its two sides along the ridge; tilt 27 deg (measured
   on property 10572052 from roof shading) and each side's real compass direction.
   Flat roofs get panels tilted 10 deg to the south.
3. Whole-year sunlight on every side at this latitude/longitude (``backend/sun.py``:
   sun every 10 min, NASA monthly weather). Sides under 80% of the best are skipped.
4. 400 W panels (1.13 x 1.72 m) fitted with a 3 ft fire setback.
5. Yearly kWh from NREL PVWatts for each side's tilt and direction.
6. Money: cost $2.00/W (range $1.80-2.40), savings = kWh x local power price
   (range +/-10%), simple payback, and savings per apartment per month.
7. Owner: CoStar TrueOwner -> its sale record's buyer name, else the company record.

Writes ``frontend/solar/data_<city>.json`` (no keys, no Vexcel imagery).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapely import wkt as shapely_wkt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Point, Polygon, box  # noqa: E402

from backend import config, sun, vexcel_detect  # noqa: E402
from backend.samples import sample_buildings  # noqa: E402
from backend.tools import costar_data, geometry, solar  # noqa: E402
from scripts.measure_sample import lot_sizes  # noqa: E402
from scripts.roof_solar_view import (  # noqa: E402
    KEEP_PERCENT, PANEL_TYPES, PERFORMANCE_RATIO, fit_panels, oblique_image, ortho_image, ridge_axis, winter_row_gap,
    pvwatts_kwh_per_kw, roof_sides, unit,
)

M_PER_DEG_LAT = 111_320.0
SQFT_TO_M2 = 0.09290304
VIEWS = ("north", "east", "south", "west")  # Vexcel oblique photo directions
VIEWS_FOR_TOP = 5  # fetch side photos for the best buildings only (Vexcel calls)
VIEW_DIR = PROJECT_ROOT / "frontend" / "solar" / "views"  # git-ignored: Vexcel imagery stays local
PITCHED_TILT_DEG = 27.0  # measured on 10572052 (6 roofs agree, 26-28 deg)
FLAT_TILT_DEG = float(solar.PANEL_TILT_DEG)  # racked panels on flat roofs
PANEL = PANEL_TYPES["400w"]  # set by --panel; every count, kW and drawing uses its real size
COST_PER_W = (1.80, 2.00, 2.40)  # low, mid, high installed cost, USD per watt
SAVINGS_SPREAD = 0.10  # +/-10% on yearly savings (model vs NREL differ ~9%)
PANELS_FOR_TOP = 0  # panels are drawn in the 3D view; the map shows roof sides only (file size)
CITY_MARKET = {"houston": 14, "los_angeles": 20}
CITY_NAME = {"houston": "Houston", "los_angeles": "Los Angeles"}


# ------------------------------------------------------------------ owners

def _jsonl(name: str):
    with open(config.COSTAR_DATA_DIR / name, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def owners(ids: set[int]) -> dict[int, dict[str, str | None]]:
    """{property id: {owner, manager}} from CoStar contacts, sale records and companies."""
    props = {r["id"]: r for r in _jsonl("commercial-property.jsonl") if r["id"] in ids}
    comp_ids = {co["compId"] for r in props.values() for c in r.get("contacts") or []
                if c.get("contactRole") == 7 for co in c.get("compOwners") or []}
    comps = {d["id"]: d for d in _jsonl("commercial-sale-comp.jsonl") if d["id"] in comp_ids}
    loc_ids = {c.get("locationId") for r in props.values() for c in r.get("contacts") or []}
    loc_ids |= {c.get("locationId") for d in comps.values() for c in d.get("contacts") or []}
    companies = {d["id"]: d.get("name") for d in _jsonl("company-detail.jsonl") if d["id"] in loc_ids}

    def person(c: dict[str, Any]) -> str | None:
        name = c.get("locationName") or companies.get(c.get("locationId"))
        if not name:
            name = " ".join(x for x in (c.get("contactFirstName"), c.get("contactLastName")) if x) or None
        return name

    out: dict[int, dict[str, str | None]] = {}
    for pid, r in props.items():
        owner = manager = None
        for c in r.get("contacts") or []:
            if c.get("contactRole") == 7 and not owner:
                for co in c.get("compOwners") or []:
                    buyers = [x for x in (comps.get(co["compId"]) or {}).get("contacts") or []
                              if x.get("contactRole") in (43, 42)]  # TrueBuyer, RecordedBuyer
                    names = [n for n in (person(b) for b in sorted(buyers, key=lambda b: -b["contactRole"])) if n]
                    if names:
                        owner = names[0]
                        break
                owner = owner or companies.get(c.get("locationId"))
            if c.get("contactRole") == 4 and not manager:
                manager = companies.get(c.get("locationId"))
        if not owner:
            owner = next((companies.get(c.get("locationId")) for c in r.get("contacts") or []
                          if c.get("contactRole") == 3 and companies.get(c.get("locationId"))), None)
        out[pid] = {"owner": owner, "manager": manager}
    return out


# ------------------------------------------------------------------ geometry

class Local:
    """Metres around a reference point <-> lon/lat."""

    def __init__(self, lat: float, lon: float) -> None:
        self.lat0, self.lon0 = lat, lon
        self.kx = M_PER_DEG_LAT * math.cos(math.radians(lat))

    def poly(self, geom_wkt: str) -> Polygon:
        g = shapely_wkt.loads(geom_wkt)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        return Polygon([((x - self.lon0) * self.kx, (y - self.lat0) * M_PER_DEG_LAT) for x, y in g.exterior.coords])

    def xy(self, lon: float, lat: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * M_PER_DEG_LAT

    def latlon(self, poly: Polygon) -> list[list[float]]:
        return [[round(self.lat0 + y / M_PER_DEG_LAT, 7), round(self.lon0 + x / self.kx, 7)]
                for x, y in poly.exterior.coords]


# ------------------------------------------------------------------ one building

def analyze(b, props: dict[str, Any], steps: list[dict[str, Any]], best_year: float,
            price: float) -> dict[str, Any] | None:
    """Solar on every roof side of one property; None if Vexcel has no outlines."""
    structures = [s for s in props.get("structures") or [] if s.get("footprint.wkt")]
    if not structures:
        return None
    loc = Local(b.latitude, b.longitude)
    sides_out = []
    ours_kwh = 0.0  # the same panels, valued with our own sun model instead of NREL's
    monthly = [0.0] * 12  # kWh per month, NREL yearly total split by our monthly sun shape
    roof_m2 = 0.0
    structs = []  # kept for the 3D scene, removed before saving
    for si, st in enumerate(structures):
        poly = loc.poly(st["footprint.wkt"])
        roof_m2 += poly.area
        structs.append({"poly": poly, "shape": st.get("roof-characteristics.shape.value"),
                        "tree_pct": _num(st.get("roof-characteristics.tree-cover.percent")),
                        "ac_units": _num(st.get("roof-objects.ac-unit.count")),
                        "solar_m2": _num(st.get("roof-objects.solar-panel-photovoltaic.area"))})
        shape = st.get("roof-characteristics.shape.value")
        if shape in ("gable", "hip"):
            sides, tilt = roof_sides(poly, shape), PITCHED_TILT_DEG
        else:
            sides, tilt = [{"poly": poly, "facing_deg": 180.0}], FLAT_TILT_DEG
        flat = tilt == FLAT_TILT_DEG
        for sd in sides:
            if flat:  # pick the rack direction that makes the most power (see flat_layout)
                fit, sd["facing_deg"] = flat_layout(sd["poly"], b.latitude, b.longitude)
            plane = sun.plane_sunlight(steps, tilt, sd["facing_deg"])
            year = plane["kwh_m2_year"]
            pct = round(100 * year / best_year)
            if not flat:
                fit = fit_panels(sd["poly"], sd["facing_deg"], tilt, PANEL)
            layout = {"orientation": fit["orientation"], "rows": fit["rows"], "row_gap_m": fit["row_gap_m"], "flat": flat}
            if pct < KEEP_PERCENT:
                # What panels here WOULD make, to show why we skip this side.
                bad = fit["panels"]
                bad_per_kw = pvwatts_kwh_per_kw(b.latitude, b.longitude, tilt, round(sd["facing_deg"] / 10) * 10) \
                    or year * PERFORMANCE_RATIO
                sides_out.append({"facing": sd["facing_deg"], "tilt": tilt, "pct": pct, "panels": 0, "kw": 0,
                                  "kwh": 0, "area_m2": round(sd["poly"].area), "layout": layout,
                                  "outline": loc.latlon(sd["poly"]), "panel_shapes": [],
                                  "skipped": {"panels": len(bad), "kw": round(len(bad) * PANEL["kw"], 1),
                                              "kwh": round(len(bad) * PANEL["kw"] * bad_per_kw),
                                              "kwh_per_panel": round(PANEL["kw"] * bad_per_kw)},
                                  "_st": si, "_poly": sd["poly"], "_eave": sd.get("eave"), "_panels": [], "_bad": bad})
                continue
            panels = fit["panels"]
            kw = len(panels) * PANEL["kw"]
            per_kw = pvwatts_kwh_per_kw(b.latitude, b.longitude, tilt, round(sd["facing_deg"] / 10) * 10)
            ours_per_kw = year * PERFORMANCE_RATIO  # 1 kW of panels = 1 kW per m2 of sun at test conditions
            if per_kw is None:  # no NREL answer: fall back to our own model
                per_kw = ours_per_kw
            ours_kwh += kw * ours_per_kw
            for m, v in enumerate(plane["monthly_kwh_m2"]):
                monthly[m] += kw * per_kw * v / year if year else 0
            sides_out.append({"facing": sd["facing_deg"], "tilt": tilt, "pct": pct, "panels": len(panels),
                              "kw": round(kw, 1), "kwh": round(kw * per_kw), "area_m2": round(sd["poly"].area),
                              "layout": layout,
                              "outline": loc.latlon(sd["poly"]),
                              "panel_shapes": [loc.latlon(p) for p in panels],
                              "_st": si, "_poly": sd["poly"], "_eave": sd.get("eave"), "_panels": panels, "_ours": kw * ours_per_kw})
    ac_lost = make_room_for_ac(sides_out, structs, b.latitude, b.longitude)
    kw = sum(s["kw"] for s in sides_out)
    kwh = sum(s["kwh"] for s in sides_out)
    if kw <= 0:
        return None
    ours_kwh = sum(s.get("_ours", 0) for s in sides_out)
    monthly = [v * kwh / ac_lost["kwh_before"] for v in monthly] if ac_lost["kwh_before"] else monthly
    n_panels = sum(s["panels"] for s in sides_out)
    costar_roof_m2 = (geometry.roof_footprint_sq_ft(b) or 0) * SQFT_TO_M2
    checks = {
        # Two independent ways to get the same yearly kWh: NREL's PVWatts (its own weather
        # station data and model) vs our sun model (NOAA sun path + NASA satellite weather).
        "nrel_kwh": round(kwh), "our_kwh": round(ours_kwh),
        "model_gap_pct": round(100 * (ours_kwh - kwh) / kwh, 1),
        # Roof size: Vexcel traced outlines vs what CoStar's records imply (one main building only).
        "vexcel_roof_m2": round(roof_m2), "costar_roof_m2": round(costar_roof_m2) or None,
        # Share of the roof covered by panels (fire gaps, wrong-facing sides and row gaps are empty).
        "panel_cover_pct": round(100 * sum(s["panels"] * PANEL["length_m"] * PANEL["width_m"] * math.cos(math.radians(s["tilt"]))
                                            for s in sides_out) / roof_m2) if roof_m2 else None,  # both seen from above
        "kwh_per_kw": round(kwh / kw),
        "capacity_factor_pct": round(100 * kwh / (kw * 8760), 1),
        "kwh_per_panel": round(kwh / n_panels),
    }
    savings = kwh * price
    cost = [kw * 1000 * c for c in COST_PER_W]
    sav = (savings * (1 - SAVINGS_SPREAD), savings, savings * (1 + SAVINGS_SPREAD))
    units = b.total_num_units
    return {
        "id": b.id, "name": b.name, "lat": b.latitude, "lon": b.longitude,
        "units": units, "max_rent": b.max_rent, "buildings_on_site": len(structures),
        "panels": sum(s["panels"] for s in sides_out), "kw": round(kw), "kwh_year": round(kwh),
        "savings_year": round(savings), "savings_range": [round(sav[0]), round(sav[2])],
        "cost": round(cost[1]), "cost_range": [round(cost[0]), round(cost[2])],
        "payback_years": round(cost[1] / savings, 1),
        "payback_range": [round(cost[0] / sav[2], 1), round(cost[2] / sav[0], 1)],
        "per_unit_month": round(savings / units / 12) if units else None,
        "roof_sides_used": sum(1 for s in sides_out if s["panels"]), "roof_sides": len(sides_out),
        "directions": directions(sides_out), "monthly_kwh": [round(v) for v in monthly],
        "checks": checks, "sides": sides_out, "_structs": structs,
        "ac_room": {"ac_units": ac_lost["ac_units"], "panels_removed": ac_lost["panels_removed"],
                    "panels_per_ac_unit": AC_PANELS},
        "skipped": skipped_summary(sides_out, kwh, n_panels, price),
        "roof_objects": {
            "ac_units": int(sum(x["ac_units"] or 0 for x in structs)),
            "tree_cover_pct": round(sum((x["tree_pct"] or 0) * x["poly"].area for x in structs) / roof_m2, 1) if roof_m2 else None,
            "existing_solar_m2": round(sum(x["solar_m2"] or 0 for x in structs)),
        },
    }


def flat_layout(poly: Polygon, lat: float, lon: float) -> tuple[dict[str, Any], float]:
    """Flat roof: rack panels toward true south, or line the rows up with the roof's own edges
    (whichever edge faces closest to south). Lined-up rows usually fit more panels; true south
    gets a little more sun each. Keep whichever makes more power a year (NREL per-kW yield)."""
    axis = ridge_axis(poly)[0]
    edge_dirs = [math.degrees(math.atan2(x, y)) % 360 for x, y in (axis, -axis, (axis[1], -axis[0]), (-axis[1], axis[0]))]
    aligned = min(edge_dirs, key=lambda d: abs(((d - 180) + 180) % 360 - 180))
    best, best_kwh, best_face = None, -1.0, 180.0
    for face in {180.0, round(aligned, 1)}:
        gap = winter_row_gap(lat, FLAT_TILT_DEG, PANEL["length_m"], face)
        fit = fit_panels(poly, face, FLAT_TILT_DEG, PANEL, row_gap_m=gap)
        per_kw = pvwatts_kwh_per_kw(lat, lon, FLAT_TILT_DEG, round(face / 10) * 10) or 1.0
        kwh = len(fit["panels"]) * PANEL["kw"] * per_kw
        if kwh > best_kwh:
            best, best_kwh, best_face = fit, kwh, face
    return best, best_face


def skipped_summary(sides: list[dict[str, Any]], kwh: float, n: int, price: float) -> dict[str, Any] | None:
    """Panels on the shady sides we skipped: how little they would make and how slowly they'd pay back."""
    bad = [s["skipped"] for s in sides if s.get("skipped") and s["skipped"]["panels"]]
    if not bad:
        return None
    p = sum(x["panels"] for x in bad)
    k = sum(x["kwh"] for x in bad)
    kw = p * PANEL["kw"]
    return {"panels": p, "kwh": k, "kwh_per_panel": round(k / p), "good_kwh_per_panel": round(kwh / n) if n else None,
            "cost": round(kw * 1000 * COST_PER_W[1]), "savings_year": round(k * price),
            "payback_years": round(kw * 1000 * COST_PER_W[1] / (k * price), 1) if k else None}


AC_PANELS = 2  # assumption: each rooftop AC unit plus its service space takes about 2 panels' room


def make_room_for_ac(sides: list[dict[str, Any]], structs: list[dict[str, Any]], lat: float, lon: float) -> dict[str, Any]:
    """Remove panels for the AC units Vexcel counts on each roof. Vexcel gives the count, not
    the spot, so panels are thinned evenly across that building's sunny sides."""
    before = sum(s["kwh"] for s in sides)
    removed = units = 0
    for si, st in enumerate(structs):
        ac = int(st["ac_units"] or 0)
        mine = [s for s in sides if s["_st"] == si and s["panels"]]
        have = sum(s["panels"] for s in mine)
        if not ac or not have:
            continue
        units += ac
        drop = min(have, ac * AC_PANELS)
        for s in mine:
            n = s["panels"]
            d = round(drop * n / have)
            keep = max(0, n - d)
            idx = set(np.linspace(0, n - 1, d).round().astype(int).tolist()) if d else set()
            s["_panels"] = [p for i, p in enumerate(s["_panels"]) if i not in idx]
            s["panel_shapes"] = [p for i, p in enumerate(s["panel_shapes"]) if i not in idx]
            keep = len(s["_panels"])
            f = keep / n
            removed += n - keep
            s["panels"], s["kw"], s["kwh"] = keep, round(s["kw"] * f, 1), round(s["kwh"] * f)
            s["_ours"] = s.get("_ours", 0) * f
    return {"ac_units": units, "panels_removed": removed, "kwh_before": before}


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


TZ_HOURS = {"houston": -6, "los_angeles": -8}  # local standard time (no daylight saving)


def sky_export(steps, city: str, best_tilt: float, best_year: float) -> dict[str, Any]:
    """Every sun position of the model year, so the page can run the same math live:
    [month, sun height deg, sun compass deg, direct W/m2, sky W/m2, total W/m2, hours it stands for, local minute]."""
    tz = TZ_HOURS[city] * 60
    rows = [[st["month"], round(st["elevation_deg"], 2), round(st["azimuth_deg"], 2), round(st["dni"], 1),
             round(st["dhi"], 1), round(st["ghi"], 1), round(st["hours"], 4), (st["minute_utc"] + tz) % 1440] for st in steps]
    return {"steps": rows, "step_minutes": 10, "best_tilt_deg": best_tilt, "best_kwh_m2_year": best_year,
            "performance_ratio": PERFORMANCE_RATIO, "albedo": 0.2, "time_label": "local standard time"}


def nrel_check(lat: float, lon: float, steps, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Our sun model vs NREL PVWatts for 24 panel angles at this city: does ours rank angles the same way?"""
    rows = []
    for tilt in (10, 25, 40):
        for face in range(0, 360, 45):
            nrel = pvwatts_kwh_per_kw(lat, lon, tilt, face)
            ours = sun.plane_sunlight(steps, tilt, face)["kwh_m2_year"] * PERFORMANCE_RATIO
            if nrel:
                rows.append({"tilt": tilt, "facing": face, "nrel": round(nrel), "ours": round(ours)})
    if not rows:
        return {}
    k = sorted(r["nrel"] / r["ours"] for r in rows)[len(rows) // 2]  # one city-wide scale (heat/inverter losses)
    for r in rows:
        r["ours_scaled"] = round(r["ours"] * k)
        r["diff_pct"] = round(100 * (r["ours_scaled"] - r["nrel"]) / r["nrel"], 1)
    rank = lambda key: [i for i, _ in sorted(enumerate(rows), key=lambda x: -x[1][key])]
    a, b = rank("nrel"), rank("ours")
    n = len(rows)
    ra, rb = {v: i for i, v in enumerate(a)}, {v: i for i, v in enumerate(b)}
    rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    return {"rows": rows, "scale": round(k, 3), "max_abs_diff_pct": max(abs(r["diff_pct"]) for r in rows),
            "median_abs_diff_pct": sorted(abs(r["diff_pct"]) for r in rows)[n // 2], "rank_agreement": round(rho, 3),
            "same_best": rows[a[0]]["tilt"] == rows[b[0]]["tilt"] and rows[a[0]]["facing"] == rows[b[0]]["facing"],
            "best_nrel": {"tilt": rows[a[0]]["tilt"], "facing": rows[a[0]]["facing"]},
            "best_ours": {"tilt": rows[b[0]]["tilt"], "facing": rows[b[0]]["facing"]}}


def sun_grid(steps, best_year: float, results: list[dict[str, Any]], price: float) -> dict[str, Any]:
    """Share of the best sun for a panel facing any way at any tilt, here. kWh per kW uses our sun
    model scaled by the city's median gap to NREL, so it lines up with the NREL numbers we quote."""
    gaps = sorted(r["checks"]["model_gap_pct"] for r in results)
    to_nrel = 1 / (1 + gaps[len(gaps) // 2] / 100) if gaps else 1
    facings, tilts = list(range(0, 360, 15)), list(range(0, 65, 5))
    pct, kwh = [], []
    for t in tilts:
        row = [sun.plane_sunlight(steps, t, f)["kwh_m2_year"] for f in facings]
        pct.append([round(100 * v / best_year) for v in row])
        kwh.append([round(v * PERFORMANCE_RATIO * to_nrel) for v in row])
    return {"facings": facings, "tilts": tilts, "pct": pct, "kwh_per_kw": kwh,
            "cost_per_w": COST_PER_W[1], "price": price, "to_nrel": round(to_nrel, 3)}


def proof(results: list[dict[str, Any]], steps: list[dict[str, Any]], clim: dict[str, Any]) -> dict[str, Any]:
    """City-wide checks that the numbers hold up, from data we did not tune to."""
    gaps = sorted(abs(r["checks"]["model_gap_pct"]) for r in results)
    roof = [(r["checks"]["vexcel_roof_m2"], r["checks"]["costar_roof_m2"]) for r in results
            if r["checks"]["costar_roof_m2"] and r["buildings_on_site"] == 1]
    roof_gaps = sorted(abs(100 * (v - c) / c) for v, c in roof)
    flat = sun.plane_sunlight(steps, 0, 180)["kwh_m2_year"]
    nasa = clim.get("sunlight_kwh_m2_year")

    def median(xs: list[float]) -> float | None:
        return round(xs[len(xs) // 2], 1) if xs else None

    return {
        "model_vs_nrel_median_gap_pct": median(gaps),
        "model_vs_nrel_within_10_pct": sum(1 for g in gaps if g <= 10), "buildings": len(gaps),
        "sun_model_flat_kwh_m2_year": flat, "nasa_flat_kwh_m2_year": nasa,
        "roof_single_buildings": len(roof_gaps), "roof_median_gap_pct": median(roof_gaps),
        "roof_within_25_pct": sum(1 for g in roof_gaps if g <= 25),
    }


FLOOR_M = 3.0  # CoStar floors x 3 m: Vexcel's eave-height field is empty for these roofs
MAX_RISE_M = 5.0  # cap the roof rise so one long ridge over an odd-shaped roof stays believable


def top_photo(pid: int, b, parcel_wkt: str, loc: "Local") -> dict[str, Any] | None:
    """Vexcel's newest top-down photo of the parcel (+15 m), and its corners in local metres."""
    pad = 15 / M_PER_DEG_LAT
    area = shapely_wkt.loads(parcel_wkt)
    bounds = (area.bounds[0] - pad, area.bounds[1] - pad, area.bounds[2] + pad, area.bounds[3] + pad)
    path = VIEW_DIR / f"{pid}_top.jpg"
    if not path.exists():
        try:
            cols = vexcel_detect.list_collections(b.latitude, b.longitude)
            col = next(c for c in cols if c["layer"] in ("urban", "urban-r"))
            img = ortho_image(box(*bounds).wkt, col["collection"], col["layer"])
        except Exception as exc:
            print(f"top photo {pid} unavailable ({type(exc).__name__})")
            return None
        img.thumbnail((2048, 2048))
        VIEW_DIR.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=88)
    (x0, y0), (x1, y1) = loc.xy(bounds[0], bounds[1]), loc.xy(bounds[2], bounds[3])
    return {"src": f"views/{pid}_top.jpg", "x0": round(x0, 2), "y0": round(y0, 2), "x1": round(x1, 2), "y1": round(y1, 2)}


def scene(r: dict[str, Any], b, parcel_wkt: str) -> dict[str, Any]:
    """A simple 3D model of the site: buildings raised from Vexcel outlines, sloped roof sides,
    and every panel at its real size and tilt. Units are metres, x east, y north, z up."""
    loc = Local(b.latitude, b.longitude)
    stories = b.num_stories_above_grade or 2
    eave = stories * FLOOR_M
    out_sides, panels, bad_panels = [], [], []
    by_struct: dict[int, list[dict[str, Any]]] = {}
    for sd in r["sides"]:
        by_struct.setdefault(sd["_st"], []).append(sd)
    for si, sides in by_struct.items():
        st = r["_structs"][si]
        c = np.array(st["poly"].centroid.coords[0])
        pitched = st["shape"] in ("gable", "hip") and all(sd.get("_eave") is not None for sd in sides)
        t = math.tan(math.radians(PITCHED_TILT_DEG))
        for k, sd in enumerate(sides):
            n = unit(sd["facing"])

            def z_roof(p, sd=sd) -> float:
                # Each face rises from its own eave at the roof's tilt (capped), so faces meet at hips and ridges.
                if not pitched:
                    return eave
                return eave + min(MAX_RISE_M, t * sd["_eave"].distance(Point(p)))

            pts = [[round(x, 2), round(y, 2), round(z_roof((x, y)), 2)] for x, y in list(sd["_poly"].exterior.coords)[:-1]]
            out_sides.append({"pts": pts, "pct": sd["pct"], "facing": sd["facing"], "tilt": sd["tilt"], "layout": sd["layout"],
                              "panels": sd["panels"], "kw": sd["kw"], "kwh": sd["kwh"]})
            side_i = len(out_sides) - 1
            for poly in sd.get("_bad") or []:
                corners = list(poly.exterior.coords)[:4]
                if pitched:
                    zs = [z_roof(p) + 0.12 for p in corners]
                else:
                    d = [float(np.array(p) @ n) for p in corners]
                    tf = math.tan(math.radians(sd["tilt"]))
                    zs = [eave + 0.3 + tf * (max(d) - di) for di in d]
                bad_panels.append([side_i] + [round(v, 2) for p, z in zip(corners, zs) for v in (p[0], p[1], z)])
            for poly in sd["_panels"]:
                corners = list(poly.exterior.coords)[:4]
                if pitched:
                    zs = [z_roof(p) + 0.12 for p in corners]
                else:  # racked: the panel's high edge is away from the way it faces
                    d = [float(np.array(p) @ n) for p in corners]
                    tf = math.tan(math.radians(sd["tilt"]))
                    zs = [eave + 0.3 + tf * (max(d) - di) for di in d]
                panels.append([side_i] + [round(v, 2) for p, z in zip(corners, zs) for v in (p[0], p[1], z)])
    return {"photo": top_photo(r["id"], b, parcel_wkt, loc), "eave_m": eave, "stories": stories,
            "stories_source": "CoStar floors" if b.num_stories_above_grade else "assumed (CoStar has no floor count)",
            "sides": out_sides, "panels": panels, "bad_panels": bad_panels}


def side_photos(pid: int, parcel_wkt: str) -> list[str]:
    """Save Vexcel's four oblique (side-angle) photos of the parcel; returns the directions saved."""
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    (VIEW_DIR / ".gitignore").write_text("*\n")  # licensed imagery: never committed
    saved = []
    for d in VIEWS:
        path = VIEW_DIR / f"{pid}_{d}.jpg"
        if not path.exists():
            img = oblique_image(parcel_wkt, f"oblique-{d}")
            if img is None:
                continue
            img.thumbnail((900, 900))
            img.save(path, quality=85)
        saved.append(d)
    return saved


COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_WORD = {"N": "north", "NE": "north-east", "E": "east", "SE": "south-east", "S": "south",
                "SW": "south-west", "W": "west", "NW": "north-west"}


def directions(sides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roof sides grouped by the way they face: panels, power and how good the sun is there."""
    groups: dict[str, dict[str, Any]] = {}
    for s in sides:
        # A flat roof doesn't face anywhere: its panels are racked toward the sun.
        key = "FLAT" if s["layout"]["flat"] else COMPASS[round(s["facing"] / 45) % 8]
        word = "flat roof" if key == "FLAT" else COMPASS_WORD[key]
        g = groups.setdefault(key, {"dir": key, "word": word, "flat": key == "FLAT", "sides": 0, "panels": 0,
                                    "kw": 0.0, "kwh": 0, "area_m2": 0, "pct_sum": 0.0})
        g["sides"] += 1
        g["panels"] += s["panels"]
        g["kw"] += s["kw"]
        g["kwh"] += s["kwh"]
        g["area_m2"] += s["area_m2"]
        g["pct_sum"] += s["pct"] * s["area_m2"]
    out = []
    for key in ["FLAT"] + COMPASS:
        if key in groups:
            g = groups.pop(key)
            g["pct"] = round(g.pop("pct_sum") / g["area_m2"]) if g["area_m2"] else 0
            g["kw"] = round(g["kw"], 1)
            g["kwh_per_panel"] = round(g["kwh"] / g["panels"]) if g["panels"] else 0
            out.append(g)
    return out


# ------------------------------------------------------------------ words

def recipient(r: dict[str, Any]) -> tuple[str, str]:
    """Who the message goes to: the owner when CoStar names one, else the property manager."""
    if r.get("owner"):
        return r["owner"], "owner"
    if r.get("manager"):
        return r["manager"], "property manager"
    return f"Owner of {r.get('name') or 'the property'}", ""


def email(r: dict[str, Any], city: str) -> dict[str, str]:
    who = recipient(r)[0]
    name = r.get("name") or f"your property (CoStar {r['id']})"
    per_unit = (f" That is about ${r['per_unit_month']} per apartment every month, money that can go "
                f"toward keeping rents steady for your {r['units']} households." if r.get("per_unit_month") else "")
    subject = f"Your roof at {name} could save about ${r['savings_year']:,} a year"
    sides_line = ("Its roof gets strong sun all year, enough for about " if r["roof_sides"] == 1 else
                  f"{r['roof_sides_used']} of its {r['roof_sides']} roof sides get strong sun all year, enough for about ")
    body = (
        f"Dear {who},\n\n"
        f"We studied the roofs at {name} in {CITY_NAME[city]} from aerial photos. "
        + sides_line + f"{r['panels']:,} solar panels ({r['kw']:,} kW).\n\n"
        f"Using this location's real sun path and weather, we estimate {r['kwh_year']:,} kWh a year, "
        f"saving about ${r['savings_range'][0]:,}-${r['savings_range'][1]:,} a year. At ${r['cost_range'][0]:,}-"
        f"${r['cost_range'][1]:,} to install, it pays for itself in about {r['payback_range'][0]}-"
        f"{r['payback_range'][1]} years.{per_unit}\n\n"
        "These are screening estimates; an on-site check would confirm them. "
        "Would you like a free roof assessment?\n\n"
        "Best regards,\nThe CityLab team"
    )
    return {"to": who, "subject": subject, "body": body}


def voice(r: dict[str, Any], city: str) -> str:
    name = r.get("name") or "this property"
    renters = (f" That's about {r['per_unit_month']} dollars per apartment, every month: money that can keep "
               f"rent steady for {r['units']} families." if r.get("per_unit_month") else "")
    return (f"This is {name} in {CITY_NAME[city]}. From the sky, we found {r['roof_sides_used']} roof sides "
            f"facing the sun, with room for {r['panels']:,} solar panels. Using the real sun path and "
            f"{CITY_NAME[city]}'s weather, those panels would save about {r['savings_year']:,} dollars a year, "
            f"and pay for themselves in about {round(r['payback_years'])} years.{renters} "
            "Right now, that money is just sitting on the roof.")


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", default="houston", choices=list(CITY_MARKET))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--no-views", action="store_true", help="skip the Vexcel side photos")
    ap.add_argument("--panel", default="400w", choices=list(PANEL_TYPES), help="panel size to lay out")
    args = ap.parse_args(argv)
    global PANEL
    PANEL = PANEL_TYPES[args.panel]

    market = CITY_MARKET[args.city]
    sample = sample_buildings(args.city, 11, n=args.n)
    full = {b.id: b for b in costar_data.load_properties()}
    sample = [full.get(b.id, b) for b in sample if b.latitude is not None]
    lots = lot_sizes([b.id for b in sample])
    who = owners({b.id for b in sample})
    price = solar.PRICE_PER_KWH.get(market, solar.DEFAULT_PRICE_PER_KWH)

    lat0 = sum(b.latitude for b in sample) / len(sample)
    lon0 = sum(b.longitude for b in sample) / len(sample)
    steps = sun.sky_year(lat0, lon0)
    best_tilt = sun.sun_summary(lat0, lon0)["best_fixed_tilt_deg"]
    best_year = sun.plane_sunlight(steps, best_tilt, 180)["kwh_m2_year"]
    clim = sun.climate(lat0, lon0) or {}

    results, parcels, skipped = [], {}, {"no_lot_size": 0, "no_matching_parcel": 0, "no_roof_outlines": 0}
    for b in sample:
        if b.id not in lots:
            skipped["no_lot_size"] += 1
            continue
        try:
            parcel = vexcel_detect.property_parcel(b.latitude, b.longitude, lots[b.id])
        except Exception:
            parcel = None
        if parcel is None:
            skipped["no_matching_parcel"] += 1
            continue
        r = analyze(b, parcel["properties"], steps, best_year, price)
        if r is None:
            skipped["no_roof_outlines"] += 1
            continue
        r.update(who.get(b.id, {}))
        results.append(r)
        parcels[b.id] = parcel["wkt"]

    results.sort(key=lambda r: -r["savings_year"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["recipient"], r["recipient_role"] = recipient(r)
        r["email"] = email(r, args.city)
        r["voice"] = voice(r, args.city)
        if i >= PANELS_FOR_TOP:
            for s in r["sides"]:
                s["panel_shapes"] = []
        r["views"] = side_photos(r["id"], parcels[r["id"]]) if i < VIEWS_FOR_TOP and not args.no_views else []
        if i < VIEWS_FOR_TOP:
            r["scene"] = scene(r, full.get(r["id"]) or next(x for x in sample if x.id == r["id"]), parcels[r["id"]])
    for r in results:
        r.pop("_structs", None)
        for sd in r["sides"]:
            for k in ("_st", "_poly", "_eave", "_panels", "_ours", "_bad"):
                sd.pop(k, None)

    for r in results:  # smaller file: 6 decimals is ~10 cm
        for sd in r["sides"]:
            sd["outline"] = [[round(a, 6), round(b2, 6)] for a, b2 in sd["outline"]]
            sd["panel_shapes"] = [[[round(a, 6), round(b2, 6)] for a, b2 in ps[:4]] for ps in sd["panel_shapes"]]
    out = {
        "city": CITY_NAME[args.city], "market_id": market,
        "assumptions": {
            "pitched_roof_tilt_deg": PITCHED_TILT_DEG, "flat_roof_panel_tilt_deg": FLAT_TILT_DEG,
            "panel": {**PANEL, "efficiency_pct": round(100 * PANEL["kw"] / (PANEL["length_m"] * PANEL["width_m"]), 1)},
            "fire_setback_ft": 3, "keep_sides_with_sun_pct": KEEP_PERCENT,
            "row_gap_rule": "flat-roof rows spaced so none shades the next from 10 am to 2 pm on Dec 21",
            "installed_cost_usd_per_w": list(COST_PER_W), "power_price_usd_per_kwh": price,
            "savings_spread": SAVINGS_SPREAD, "incentives": "none assumed",
        },
        "weather": {"cloud_cover_percent": clim.get("cloud_cover_percent"),
                    "cloud_loss_percent": clim.get("cloud_loss_percent"),
                    "sunlight_kwh_m2_year": clim.get("sunlight_kwh_m2_year"),
                    "temperature_c": clim.get("temperature_c"),
                    "source": clim.get("source"),
                    "monthly": [{k: m.get(k) for k in ("month", "sunlight_kwh_m2_day", "clear_sky_kwh_m2_day",
                                                       "cloud_cover_percent", "temperature_c")}
                                for m in clim.get("monthly") or []]},
        "proof": proof(results, steps, clim),
        "sun_grid": sun_grid(steps, best_year, results, price),
        "sky": sky_export(steps, args.city, best_tilt, best_year),
        "nrel_check": nrel_check(lat0, lon0, steps, results),
        "sun": {"paths": {k: sorted(([round(q["azimuth_deg"], 1), round(q["elevation_deg"], 1)]
                                     for q in sun.sun_path(lat0, lon0, date(2026, m, 21), 20)), key=lambda a: a[0])
                          for k, m in (("summer", 6), ("spring_fall", 3), ("winter", 12))},
                "latitude": round(lat0, 3), "longitude": round(lon0, 3), "positions": len(steps),
                "best_tilt_deg": best_tilt, "best_kwh_m2_year": best_year},
        "counts": {"buildings_in_sample": len(sample), "analyzed": len(results),
                   "roofs": sum(r["buildings_on_site"] for r in results),
                   "roof_sides": sum(r["roof_sides"] for r in results),
                   "panels": sum(r["panels"] for r in results),
                   "savings_year": sum(r["savings_year"] for r in results), **skipped},
        "buildings": results,
    }
    path = PROJECT_ROOT / "frontend" / "solar" / f"data_{args.city}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(out["counts"]))
    for r in results[:5]:
        print(r["rank"], r["id"], r.get("name"), "| to:", r.get("recipient"), "|", r["panels"], "panels |",
              f"${r['savings_year']:,}/yr | payback {r['payback_range']} yrs | ${r.get('per_unit_month')}/unit/mo")
    print("recipients:", {k: sum(1 for r in results if r["recipient_role"] == k) for k in
                          ("owner", "property manager", "")})
    print(f"saved: {path} ({path.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    main()
