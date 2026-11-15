"""Read the committed showcase samples in ``data/samples/`` as Buildings.

The CSVs are built by ``scripts/prepare_showcase.py`` (seeded, reproducible),
so sampling works without the CoStar kit and without network access. This is
a data helper, not an agent tool; the agent-facing loader is
``backend.tools.costar_data``.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from backend import config
from backend.schemas import Building

SAMPLES_DIR = config.PROJECT_ROOT / "data" / "samples"
CITY_FILES = {
    "houston": "houston",
    "los angeles": "los_angeles",
    "los_angeles": "los_angeles",
    "la": "los_angeles",
}
_INT_FIELDS = ("market_id", "property_type", "total_num_units", "year_built",
               "building_class", "surface_parking_spaces", "num_stories_above_grade",
               "lot_size_measurement_type", "building_construction_material", "last_sale_price_status")
_FLOAT_FIELDS = ("latitude", "longitude", "max_rent", "vacancy_percent",
                  "gross_typical_floor_sq_ft", "building_height_ft", "lot_size",
                  "last_sale_price", "last_sale_price_per_unit")


def _value(raw: object, as_int: bool) -> int | float | None:
    """Return a finite number (int when asked), or None for a blank cell."""
    try:
        f = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return int(f) if as_int else f


def _floor_areas(raw: object) -> list[float] | None:
    """Parse the CSV's JSON-encoded per-floor area list, or None for a blank cell."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        areas = json.loads(raw)
    except ValueError:
        return None
    return [float(a) for a in areas] if isinstance(areas, list) and areas else None


def sample_buildings(
    city: str, property_type: int = 11, outcome: str | None = None, n: int = 50,
) -> list[Building]:
    """Return up to *n* sampled Buildings for *city*, sorted by id.

    Args:
        city: ``"houston"`` or ``"los_angeles"`` (``"Los Angeles"``/``"LA"`` work).
        property_type: CoStar ``propertyType``; Multifamily is 11.
        outcome: Optional Building field that must be known, e.g. ``"max_rent"``.
            Blank means unknown, so those rows are dropped.
        n: Maximum number of buildings.
    """
    slug = CITY_FILES.get(city.strip().lower())
    if slug is None:
        raise ValueError(f"No sample for city {city!r}; known: houston, los_angeles")
    path = SAMPLES_DIR / f"{slug}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/prepare_showcase.py")
    if outcome is not None and outcome not in Building.model_fields:
        raise ValueError(f"Unknown outcome {outcome!r}; use a Building field name")

    df = pd.read_csv(path)
    df = df[df["property_type"] == property_type].sort_values("property_id")

    buildings: list[Building] = []
    for row in df.to_dict(orient="records"):
        fields = {k: _value(row.get(k), True) for k in _INT_FIELDS}
        fields.update({k: _value(row.get(k), False) for k in _FLOAT_FIELDS})
        fields["floor_areas_sq_ft"] = _floor_areas(row.get("floor_areas_sq_ft"))
        sale_date = row.get("last_sale_date")
        fields["last_sale_date"] = sale_date if isinstance(sale_date, str) and sale_date.strip() else None
        building = Building(id=int(row["property_id"]), **fields)
        if outcome is None or getattr(building, outcome) is not None:
            buildings.append(building)
        if len(buildings) >= max(n, 0):
            break
    return buildings[: max(n, 0)]
