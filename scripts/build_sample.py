"""Build the showcase Q1 sample: 40 to 50 Houston buildings, Vexcel data for a few.

Run ``python scripts/build_sample.py`` (add ``--vexcel 0`` to skip Vexcel).
Buildings come from the committed ``data/samples/houston.csv``; Vexcel
attributes and aerial images go through the cache, so re-running is cheap and
``USE_CACHE_ONLY=true`` replays them offline. A Vexcel failure on one building
is recorded in that building's entry and the script keeps going.

Output: ``data/cache/sample/<city>_q1.json`` (git-ignored). The console shows
only counts and the output path, never keys or response bodies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config  # noqa: E402
from backend.tools import costar_data, vexcel  # noqa: E402

Q1_OUTCOME = "max_rent"
# Vexcel precomputed fields worth keeping next to each building; the full
# response stays in the Vexcel cache.
KEEP_FIELDS = (
    "parcel-tree-cover.percent",
    "hardscapes.percent",
    "hardscapes.area",
    "pool.detected",
    "pool.total-area",
    "structure.count",
    "metadata.ortho-date-estimate",
)


def summarize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep tree cover, paved area, pool and structure count (None = unknown)."""
    return {key: attributes.get(key) for key in KEEP_FIELDS}


def add_vexcel(building: dict[str, Any]) -> dict[str, Any]:
    """Return Vexcel precomputed attributes and the aerial image path for one building."""
    lat, lon = building["latitude"], building["longitude"]
    info = vexcel.get_precomputed(lat, lon)
    aerial = vexcel.get_aerial(lat, lon, building["id"])
    return {
        "attributes": summarize_attributes(info.get("attributes") or {}),
        "image_path": aerial["image_path"],
    }


def build_sample(
    city: str = "houston", n: int = 50, vexcel_count: int = 3,
) -> dict[str, Any]:
    """Load the sample and attach Vexcel data to the first *vexcel_count* buildings."""
    buildings = costar_data.sample_buildings(city, 11, outcome=Q1_OUTCOME, n=n)
    for i, building in enumerate(buildings):
        if i >= vexcel_count:
            building["vexcel"] = None
            continue
        try:
            building["vexcel"] = add_vexcel(building)
        except Exception as exc:  # one bad building must not stop the run
            building["vexcel"] = {"error": type(exc).__name__}
    return {
        "city": city,
        "question_id": "Q1",
        "outcome": Q1_OUTCOME,
        "n": len(buildings),
        "buildings": buildings,
    }


def main(argv: list[str] | None = None) -> Path:
    """Build the sample, save it and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", default="houston")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--vexcel", type=int, default=3, help="buildings to fetch Vexcel for")
    args = parser.parse_args(argv)

    sample = build_sample(args.city, args.n, args.vexcel)
    out = config.CACHE_DIR / "sample" / f"{args.city}_q1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")

    with_vexcel = sum(1 for b in sample["buildings"] if b["vexcel"] and "error" not in b["vexcel"])
    failed = sum(1 for b in sample["buildings"] if b["vexcel"] and "error" in b["vexcel"])
    print(f"buildings: {sample['n']}  with Vexcel: {with_vexcel}  Vexcel errors: {failed}")
    print(f"saved: {out}")
    return out


if __name__ == "__main__":
    main()
