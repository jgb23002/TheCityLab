"""Local CoStar JSON Lines interfaces; there is no CoStar API.

Reads ``data/costar/data/*.jsonl`` with ``pandas.read_json(lines=True)`` and
returns :class:`backend.schemas.Building` models (see docs/CONTRACTS.md for the
field mappings). Unknown values stay ``None``, never zero.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from backend import config
from backend.schemas import Building

_PROPERTIES_FILE = "commercial-property.jsonl"
_APARTMENT_FILE = "apartment-listing.jsonl"
_MATTERPORT_CSV = "commercial-properties.csv"
_SURFACE_PARKING_TYPES = (1, 2)  # Surface and SurfaceTandem


# ---------------------------------------------------------------------------
# Loading (cached per file path: the property file is ~190 MB)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read one JSONL file once per process."""
    return pd.read_json(path, lines=True)


def _load_jsonl(filename: str) -> pd.DataFrame:
    """Read a JSONL file from the CoStar data directory."""
    path = config.COSTAR_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"CoStar data file not found: {path}. Unzip the CoStar kit into data/costar/ first."
        )
    return _read_jsonl(path)


def _matterport_map() -> dict[int, str]:
    """Return {PropertyId: MatterportModelId} from the kit CSV (non-blank only)."""
    csv_path = config.COSTAR_DATA_DIR.parent / _MATTERPORT_CSV
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype={"MatterportModelId": str})
    result: dict[int, str] = {}
    for pid, mid in zip(df["PropertyId"], df["MatterportModelId"]):
        if pd.notna(pid) and isinstance(mid, str) and mid.strip():
            result[int(pid)] = mid.strip()
    return result


def _max_rent_map() -> dict[int, float]:
    """Return {propertyId: maxRent} from apartment listings; zero means unreported."""
    path = config.COSTAR_DATA_DIR / _APARTMENT_FILE
    if not path.exists():
        return {}
    df = _read_jsonl(path)
    result: dict[int, float] = {}
    for pid, rent in zip(df["propertyId"], df["maxRent"]):
        value = _finite(rent)
        if pd.notna(pid) and value is not None and value > 0:
            result[int(pid)] = value
    return result


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def _finite(value: object) -> float | None:
    """Return a finite float, or None for missing / infinite / NaN / bool."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _whole(value: object) -> int | None:
    """Return a non-negative whole number as int, else None."""
    f = _finite(value)
    return int(f) if f is not None and f >= 0 and f.is_integer() else None


def _earliest_completion_year(events: object) -> int | None:
    """Return the earliest ConstructionComplete (eventType == 2) year."""
    if not isinstance(events, list):
        return None
    years = [
        y for ev in events
        if isinstance(ev, dict) and ev.get("eventType") == 2
        and (y := _whole(ev.get("year"))) is not None and y > 0
    ]
    return min(years) if years else None


def _surface_parking_spaces(parking: object) -> int | None:
    """Sum Surface + SurfaceTandem spaces; absent or incomplete counts are unknown."""
    if not isinstance(parking, list):
        return None
    entries = [
        e for e in parking
        if isinstance(e, dict) and e.get("parkingType") in _SURFACE_PARKING_TYPES
    ]
    if not entries:
        return None
    counts = [_whole(e.get("totalSpaces")) for e in entries]
    return None if any(c is None for c in counts) else sum(counts)  # type: ignore[arg-type]


def _to_building(row: dict[str, Any], mp_map: dict[int, str], rents: dict[int, float]) -> Building:
    """Convert one commercial-property record into a Building."""
    pid = int(row["id"])
    addr = row.get("primaryAddress")
    addr = addr if isinstance(addr, dict) else {}
    apt = row.get("propertyApartment")
    apt = apt if isinstance(apt, dict) else {}

    lat, lon = _finite(addr.get("latitude")), _finite(addr.get("longitude"))
    vacancy = _finite(apt.get("vacancyPercent"))
    units = _whole(apt.get("totalNumUnits"))
    name = row.get("buildingName")

    return Building(
        id=pid,
        primary_address=addr or None,
        property_type=_whole(row.get("propertyType")),
        market_id=_whole(addr.get("researchMarketId")),
        latitude=lat if lat is not None and -90 <= lat <= 90 else None,
        longitude=lon if lon is not None and -180 <= lon <= 180 else None,
        matterport_model_id=mp_map.get(pid),
        name=name if isinstance(name, str) and name.strip() else None,
        building_size_sq_ft=_finite(row.get("grossBuildingAreaSquareFeet")),
        year_built=_earliest_completion_year(row.get("constructionEvents")),
        building_class=_whole(row.get("buildingClass")),
        total_num_units=units if units else None,
        max_rent=rents.get(pid),
        vacancy_percent=vacancy if vacancy is not None and 0 <= vacancy <= 100 else None,
        surface_parking_spaces=_surface_parking_spaces(row.get("parking")),
    )


def _clean(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return records with pandas NaN placeholders replaced by None."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Public tool functions (signatures frozen in docs/CONTRACTS.md)
# ---------------------------------------------------------------------------

def load_properties() -> list[Building]:
    """Return every commercial property in the kit as a Building, sorted by id."""
    df = _load_jsonl(_PROPERTIES_FILE).sort_values("id")
    mp_map, rents = _matterport_map(), _max_rent_map()
    return [_to_building(row, mp_map, rents) for row in _clean(df)]


def get_property(id: int) -> Building | None:
    """Return the commercial property with this integer ``id``, or None."""
    df = _load_jsonl(_PROPERTIES_FILE)
    matches = df[df["id"] == id]
    if matches.empty:
        return None
    return _to_building(_clean(matches.head(1))[0], _matterport_map(), _max_rent_map())


def find_buildings(
    market_id: int, property_type: int, limit: int = 20
) -> list[Building]:
    """Return up to *limit* geocoded buildings in a market, sorted by id.

    Filters ``primaryAddress.researchMarketId`` (Houston 14, Los Angeles 20)
    and ``propertyType`` (Multifamily 11). Buildings without a valid latitude
    and longitude are skipped.
    """
    df = _load_jsonl(_PROPERTIES_FILE)
    df = df[df["propertyType"] == property_type].sort_values("id")
    mp_map, rents = _matterport_map(), _max_rent_map()

    results: list[Building] = []
    for row in _clean(df):
        if len(results) >= limit:
            break
        addr = row.get("primaryAddress")
        if not isinstance(addr, dict) or addr.get("researchMarketId") != market_id:
            continue
        building = _to_building(row, mp_map, rents)
        if building.latitude is not None and building.longitude is not None:
            results.append(building)
    return results
