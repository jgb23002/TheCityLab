"""Sun position and solar climate for every sampled building, Houston vs Los Angeles.

Run ``python scripts/solar_conditions.py`` (``--cities houston los_angeles``).

For each building in ``data/samples/<city>.csv`` it writes sun geometry (NOAA
equations) and climate (NASA POWER: sunlight, clear-sky sunlight, cloud cover,
cloud loss, temperature) to ``data/cache/solar_conditions/<city>.json``, plus
a city comparison in ``data/cache/solar_conditions/summary.json``.

NASA POWER is free and needs no key; responses are cached per ~50 km grid
cell, so a whole city takes 1-3 calls. No model is used anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config, sun  # noqa: E402
from backend.samples import sample_buildings  # noqa: E402

CLIMATE_KEYS = ("sunlight_kwh_m2_day", "sunlight_kwh_m2_year", "clear_sky_kwh_m2_day",
                "cloud_cover_percent", "cloud_loss_percent", "temperature_c")


def _median(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def city_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median sun and climate figures across a city's buildings."""
    climates = [r["climate"] for r in rows if r.get("climate")]
    suns = [r["sun"] for r in rows]
    out: dict[str, Any] = {"buildings": len(rows), "with_climate": len(climates)}
    for key in CLIMATE_KEYS:
        out[key] = _median([c.get(key) for c in climates])
    out["latitude"] = _median([s["latitude"] for s in suns])
    for season in ("summer_solstice", "winter_solstice"):
        out[f"{season}_noon_elevation_deg"] = _median([s[season]["noon_elevation_deg"] for s in suns])
        out[f"{season}_day_length_hours"] = _median([s[season]["day_length_hours"] for s in suns])
    out["best_fixed_tilt_deg"] = _median([s["best_fixed_tilt_deg"] for s in suns])
    out["grid_cells"] = sorted({tuple(c["grid_cell"]) for c in climates})
    if climates:  # monthly medians for the chart
        out["monthly"] = [
            {"month": m, **{k: _median([c["monthly"][i][k] for c in climates])
                            for k in ("sunlight_kwh_m2_day", "cloud_cover_percent")}}
            for i, m in enumerate(sun.MONTHS)
        ]
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cities", nargs="+", default=["houston", "los_angeles"])
    args = parser.parse_args(argv)

    out_dir = config.CACHE_DIR / "solar_conditions"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for city in args.cities:
        rows = []
        for b in sample_buildings(city, 11, n=1000):
            if b.latitude is None or b.longitude is None:
                continue
            rows.append({"property_id": b.id, **sun.solar_conditions(b.latitude, b.longitude)})
        (out_dir / f"{city}.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        summary[city] = city_summary(rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=list) + "\n",
                                          encoding="utf-8")
    rows = [("buildings", "buildings"), ("latitude", "latitude"),
            ("sunlight kWh/m2/day", "sunlight_kwh_m2_day"),
            ("clear-sky kWh/m2/day", "clear_sky_kwh_m2_day"),
            ("cloud cover %", "cloud_cover_percent"), ("sunlight lost to clouds %", "cloud_loss_percent"),
            ("temperature C", "temperature_c"),
            ("Jun 21 noon sun height deg", "summer_solstice_noon_elevation_deg"),
            ("Dec 21 noon sun height deg", "winter_solstice_noon_elevation_deg"),
            ("Jun 21 day length h", "summer_solstice_day_length_hours"),
            ("Dec 21 day length h", "winter_solstice_day_length_hours"),
            ("best panel tilt deg", "best_fixed_tilt_deg")]
    print(f"{'':30}" + "".join(f"{c:>14}" for c in summary))
    for label, key in rows:
        print(f"{label:30}" + "".join(f"{str(summary[c].get(key)):>14}" for c in summary))
    print(f"saved: {out_dir}")
    return summary


if __name__ == "__main__":
    main()
