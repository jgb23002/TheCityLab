"""Local CoStar JSON Lines interfaces; there is no CoStar API.

Data lives in ``data/costar/data/`` as ``.jsonl`` files.  Read
``data/costar/index.md`` and ``data/costar/models/`` before adding new field
references.  Field names use camelCase; enum values are integers.

Houston is ``primaryAddress.researchMarketId == 14``.
propertyType codes: Multifamily = 11, Office = 5, Industrial = 2.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from backend import config

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None

_PROPERTIES_FILE = "commercial-property.jsonl"
_APARTMENT_FILE = "apartment-listing.jsonl"
_MATTERPORT_CSV = "commercial-properties.csv"


def _data_dir() -> Path:
    return config.COSTAR_DATA_DIR


def _load_jsonl(filename: str) -> pd.DataFrame:
    """Read a JSONL file from the CoStar data directory."""
    path = _data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"CoStar data file not found: {path}\n"
            "Unzip the CoStar kit into data/costar/ first."
        )
    return pd.read_json(path, lines=True)


def _finite(value: object) -> float | None:
    """Return a finite float, or None for missing / infinite / NaN."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _earliest_completion_year(events: object) -> int | None:
    """Return the earliest ConstructionComplete (eventType == 2) year."""
    if not isinstance(events, list):
        return None
    years = []
    for ev in events:
        if isinstance(ev, dict) and ev.get("eventType") == 2:
            y = _finite(ev.get("year"))
            if y is not None and y > 0 and y == int(y):
                years.append(int(y))
    return min(years) if years else None


def _matterport_map() -> dict[int, str]:
    """Return {PropertyId: MatterportModelId} from the CSV (non-blank models only)."""
    csv_path = config.PROJECT_ROOT / "data" / "costar" / _MATTERPORT_CSV
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    result: dict[int, str] = {}
    for _, row in df.iterrows():
        pid = row.get("PropertyId")
        mid = str(row.get("MatterportModelId", "")).strip()
        if pd.notna(pid) and mid:
            result[int(pid)] = mid
    return result


def _row_to_dict(row: dict[str, Any], mp_map: dict[int, str]) -> dict[str, JsonValue]:
    """Convert a commercial-property record to a JSON-serialisable Building dict."""
    pid = int(row["id"])
    addr = row.get("primaryAddress") or {}
    if not isinstance(addr, dict):
        addr = {}

    lat = _finite(addr.get("latitude"))
    lon = _finite(addr.get("longitude"))

    return {
        "id": pid,
        "property_type": row.get("propertyType"),
        "market_id": addr.get("researchMarketId"),
        "latitude": lat,
        "longitude": lon,
        "primary_address": addr,
        "gross_building_area_sq_ft": _finite(row.get("grossBuildingAreaSquareFeet")),
        "year_built": _earliest_completion_year(row.get("constructionEvents")),
        "building_class": row.get("buildingClass"),
        "matterport_model_id": mp_map.get(pid),
        # Apartment-specific (None for non-multifamily)
        "total_num_units": None,
        "vacancy_percent": None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_properties() -> list[dict[str, JsonValue]]:
    """Return all commercial properties as JSON-serialisable dicts.

    Reads ``commercial-property.jsonl`` with ``pandas.read_json(lines=True)``.
    Does NOT load apartment listings; call :func:`get_property` or
    :func:`find_buildings` for enriched records.
    """
    df = _load_jsonl(_PROPERTIES_FILE)
    mp_map = _matterport_map()
    return [_row_to_dict(row, mp_map) for row in df.to_dict(orient="records")]


def get_property(property_id: int) -> dict[str, JsonValue] | None:
    """Return one commercial property by its integer ``id``, or None.

    For Multifamily properties the ``vacancy_percent`` and ``total_num_units``
    fields are populated from the property's ``propertyApartment`` sub-object.
    """
    df = _load_jsonl(_PROPERTIES_FILE)
    mp_map = _matterport_map()

    matches = df[df["id"] == property_id]
    if matches.empty:
        return None

    row = matches.iloc[0].to_dict()
    record = _row_to_dict(row, mp_map)

    apt = row.get("propertyApartment")
    if isinstance(apt, dict):
        record["total_num_units"] = _finite(apt.get("totalNumUnits"))
        record["vacancy_percent"] = _finite(apt.get("vacancyPercent"))

    return record


def find_buildings(
    market_id: int,
    property_type: int,
    limit: int = 50,
) -> list[dict[str, JsonValue]]:
    """Return up to *limit* buildings filtered by market and property type.

    Only buildings with valid latitude **and** longitude are included.
    Houston is ``market_id=14``; Multifamily is ``property_type=11``.

    For Multifamily the returned records include ``vacancy_percent`` and
    ``total_num_units`` from the ``propertyApartment`` sub-object.

    Returns records sorted by ascending ``id``.
    """
    df = _load_jsonl(_PROPERTIES_FILE)
    mp_map = _matterport_map()

    # Filter by property type first (fast integer comparison)
    df = df[df["propertyType"] == property_type]

    results = []
    for row in df.to_dict(orient="records"):
        addr = row.get("primaryAddress") or {}
        if not isinstance(addr, dict):
            continue
        if addr.get("researchMarketId") != market_id:
            continue
        lat = _finite(addr.get("latitude"))
        lon = _finite(addr.get("longitude"))
        if lat is None or lon is None:
            continue

        record = _row_to_dict(row, mp_map)

        apt = row.get("propertyApartment")
        if isinstance(apt, dict):
            record["total_num_units"] = _finite(apt.get("totalNumUnits"))
            record["vacancy_percent"] = _finite(apt.get("vacancyPercent"))

        results.append(record)

    # Sort by id ascending, then cap
    results.sort(key=lambda r: r["id"])  # type: ignore[return-value]
    return results[:limit]


def load_apartment_listings(property_ids: list[int] | None = None) -> pd.DataFrame:
    """Return apartment listings, optionally filtered to *property_ids*.

    Columns include ``propertyId``, ``minRent``, ``maxRent``.
    Zero rents are kept (distinction between zero and missing is the caller's).
    """
    df = _load_jsonl(_APARTMENT_FILE)
    if property_ids is not None:
        df = df[df["propertyId"].isin(property_ids)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-plan interface: sample_buildings (reads the committed sample CSVs)
# ---------------------------------------------------------------------------

SAMPLES_DIR = config.PROJECT_ROOT / "data" / "samples"
_CITY_ALIASES = {
    "houston": "houston",
    "los angeles": "los_angeles",
    "los_angeles": "los_angeles",
    "la": "los_angeles",
}


def sample_buildings(
    city: str,
    property_type: int = 11,
    outcome: str | None = None,
    n: int = 50,
) -> list[dict[str, JsonValue]]:
    """Return up to *n* sampled buildings for *city* from ``data/samples/``.

    The samples are built by ``scripts/prepare_showcase.py`` (seeded, so they
    are reproducible) and are committed, so this works without the CoStar kit
    and without network access.

    Args:
        city: ``"houston"`` or ``"los_angeles"`` (``"Los Angeles"``/``"LA"`` work).
        property_type: CoStar ``propertyType`` integer; Multifamily is 11.
        outcome: Optional column that must be known, e.g. ``"max_rent"``.
            Rows where it is blank are dropped (blank means unknown, not zero).
        n: Maximum number of buildings to return.

    Returns:
        Building dicts sorted by ``id`` with ``id``, ``latitude``, ``longitude``,
        ``total_num_units`` (size), ``year_built``, ``building_class`` and the
        outcome columns ``max_rent``, ``vacancy_percent`` and
        ``surface_parking_spaces``. Unknown values are ``None``.
    """
    slug = _CITY_ALIASES.get(city.strip().lower())
    if slug is None:
        raise ValueError(f"No sample for city {city!r}; known: houston, los_angeles")
    path = SAMPLES_DIR / f"{slug}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/prepare_showcase.py")

    df = pd.read_csv(path)
    df = df[df["property_type"] == property_type]
    if outcome is not None:
        if outcome not in df.columns:
            raise ValueError(f"Unknown outcome {outcome!r}; columns: {list(df.columns)}")
        df = df[df[outcome].notna()]
    df = df.sort_values("property_id").head(max(n, 0))

    int_cols = {"market_id", "property_type", "year_built", "building_class",
                "apartment_listing_id", "surface_parking_spaces"}
    results: list[dict[str, JsonValue]] = []
    for row in df.to_dict(orient="records"):
        record: dict[str, JsonValue] = {"id": int(row.pop("property_id"))}
        for key, value in row.items():
            num = _finite(value)
            record[key] = None if num is None else (int(num) if key in int_cols else num)
        results.append(record)
    return results
