"""Run the new features end to end on a few properties and cache the results.

Usage: python -m scripts.feature_demo 23453 4548418 [--no-detect] [--no-change] [--refresh]

Per property: Vexcel precomputed facts -> Vexcel detection ("parking space", "car")
-> Sky Inspector (roof colour, cars, spaces) -> lot fullness -> Empty Lot Planner
-> heat score and retrofit comparison -> change over time. Everything is cached under
data/cache/, and a summary goes to data/cache/features/<id>.json.
"""

from __future__ import annotations

import argparse
import json

from backend import config
from backend.schemas import Building, Measurement
from backend.tools import change, costar_data, heat, lot, roof_facts, vexcel, vision


def run_one(pid: int, detect: bool, compare: bool, refresh: bool) -> dict:
    """All new features for one property."""
    b: Building | None = costar_data.get_property(pid)
    if b is None or b.latitude is None:
        return {"property_id": pid, "error": "not in kit or no location"}
    info = vexcel.get_property_info(b.latitude, b.longitude)
    facts = roof_facts.roof_facts(info)
    run, image = None, None
    if detect:
        try:
            from backend import vexcel_detect

            run = vexcel_detect.detect(b.latitude, b.longitude, ["parking space", "car"], pid)
            image = run.image_path
        except Exception as exc:
            print(f"  detect failed ({type(exc).__name__}); vision will count on its own")
    if image is None:
        image = vexcel.get_aerial(b.latitude, b.longitude, pid).image_path
    cache = config.CACHE_DIR / "vision" / f"features-{pid}.json"
    if cache.exists() and not refresh:
        m = Measurement.model_validate_json(cache.read_text())
    else:
        m = vision.inspect(image, pid, run)
        if not (m.observations or {}).get("error"):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(m.model_dump_json())
    capture_date = None
    try:  # vexcel_detect/get_aerial use the newest capture, so date the photo by it
        from backend import vexcel_detect

        sharp = [c for c in vexcel_detect.list_collections(b.latitude, b.longitude) if c["layer"] in ("urban", "urban-r")]
        capture_date = sharp[0]["date"] if sharp else None
    except Exception as exc:
        print(f"  capture dates unavailable ({type(exc).__name__})")
    full = lot.lot_fullness(b, m, facts, capture_date)
    plan = lot.plan_empty_lot(b, full)
    heat_row = heat.rank_heat([b], [m], {pid: facts})[0]
    retrofits = heat.compare_retrofits(b, m, facts, plan.get("carport"))
    out = {"property_id": pid, "name": b.name, "market_id": b.market_id, "facts": facts,
           "vision": {k: (m.observations or {}).get(k) for k in ("roof_color", "cars_seen", "lot_fullness_percent", "notes", "error")}
           | {"parking_spaces": m.parking_space_count, "vexcel_boxes": (m.observations or {}).get("vexcel_box_count"),
              "vexcel_car_boxes": (m.observations or {}).get("vexcel_car_box_count")},
           "lot_fullness": full, "empty_lot_plan": plan, "heat": heat_row, "retrofit_options": retrofits}
    if compare:
        out["change"] = change.compare_dates(b, refresh=refresh)
    path = config.CACHE_DIR / "features" / f"{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv: list[str] | None = None) -> int:
    """Print a short report per property."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--no-change", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    for pid in args.ids:
        r = run_one(pid, not args.no_detect, not args.no_change, args.refresh)
        if r.get("error"):
            print(pid, r["error"]); continue
        print(f"\n=== {pid} {r['name']} ===")
        print("  vision:", r["vision"])
        print("  lot:", r["lot_fullness"]["summary"], "|", r["lot_fullness"]["flag"])
        print("  plan:", r["empty_lot_plan"].get("recommendation"))
        print("  rule:", r["empty_lot_plan"].get("rule_check", {}).get("reason"))
        print("  heat:", r["heat"]["summary"])
        print("  retrofits:", [(o["option"], o["payback_years"]) for o in r["retrofit_options"]])
        if "change" in r:
            print("  change:", r["change"].get("summary") or r["change"].get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
