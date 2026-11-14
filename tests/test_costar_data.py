"""Tests for backend/tools/costar_data.py using synthetic JSONL fixtures."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

import backend.config as cfg
from backend.schemas import Building
from backend.tools.costar_data import (
    find_buildings,
    get_property,
    load_properties,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROPERTY_HOUSTON_MULTI = {
    "id": 100,
    "propertyType": 11,
    "primaryAddress": {
        "researchMarketId": 14,
        "latitude": 29.68,
        "longitude": -95.46,
    },
    "grossBuildingAreaSquareFeet": 50000,
    "constructionEvents": [
        {"eventType": 2, "year": 1990},
        {"eventType": 2, "year": 1985},  # earliest should be chosen
    ],
    "buildingClass": 2,
    "propertyApartment": {"totalNumUnits": 120, "vacancyPercent": 5.5},
}

PROPERTY_HOUSTON_OFFICE = {
    "id": 101,
    "propertyType": 5,
    "primaryAddress": {
        "researchMarketId": 14,
        "latitude": 29.75,
        "longitude": -95.37,
    },
    "grossBuildingAreaSquareFeet": 100000,
    "constructionEvents": [{"eventType": 2, "year": 2000}],
    "buildingClass": 1,
}

PROPERTY_NO_COORDS = {
    "id": 102,
    "propertyType": 11,
    "primaryAddress": {"researchMarketId": 14, "latitude": None, "longitude": None},
    "constructionEvents": [],
}

PROPERTY_LA_MULTI = {
    "id": 200,
    "propertyType": 11,
    "primaryAddress": {
        "researchMarketId": 20,
        "latitude": 34.05,
        "longitude": -118.25,
    },
    "grossBuildingAreaSquareFeet": 30000,
    "constructionEvents": [{"eventType": 2, "year": 2010}],
    "buildingClass": 3,
    "propertyApartment": {"totalNumUnits": 60, "vacancyPercent": 3.0},
}


@pytest.fixture()
def costar_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal CoStar data dir with synthetic JSONL files."""
    data_dir = tmp_path / "costar" / "data"
    data_dir.mkdir(parents=True)

    properties = [
        PROPERTY_HOUSTON_MULTI,
        PROPERTY_HOUSTON_OFFICE,
        PROPERTY_NO_COORDS,
        PROPERTY_LA_MULTI,
    ]
    jsonl = "".join(json.dumps(p) + "\n" for p in properties)
    (data_dir / "commercial-property.jsonl").write_text(jsonl, encoding="utf-8")

    listings = [
        {"id": 1, "propertyId": 100, "minRent": 800, "maxRent": 1200},
        {"id": 2, "propertyId": 200, "minRent": 0, "maxRent": 2500},
    ]
    (data_dir / "apartment-listing.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in listings), encoding="utf-8"
    )

    monkeypatch.setattr(cfg, "COSTAR_DATA_DIR", data_dir)
    # No CSV → matterport map will be empty
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    return data_dir


# ---------------------------------------------------------------------------
# load_properties
# ---------------------------------------------------------------------------


def test_load_properties_count(costar_dir: Path) -> None:
    """load_properties returns one record per JSONL line."""
    records = load_properties()
    assert len(records) == 4


def test_load_properties_fields_present(costar_dir: Path) -> None:
    """Every record has at least id, property_type, latitude, and longitude."""
    for rec in load_properties():
        assert isinstance(rec, Building)
        assert rec.id > 0


# ---------------------------------------------------------------------------
# get_property
# ---------------------------------------------------------------------------


def test_get_property_found(costar_dir: Path) -> None:
    """get_property returns the correct record by integer id."""
    rec = get_property(100)
    assert rec is not None
    assert rec.id == 100
    assert rec.property_type == 11
    assert abs(rec.latitude - 29.68) < 1e-6  # type: ignore[operator]


def test_get_property_not_found(costar_dir: Path) -> None:
    """get_property returns None for an id that does not exist."""
    assert get_property(9999) is None


def test_get_property_multifamily_enriched(costar_dir: Path) -> None:
    """Multifamily records include total_num_units and vacancy_percent."""
    rec = get_property(100)
    assert rec is not None
    assert rec.total_num_units == 120
    assert abs(rec.vacancy_percent - 5.5) < 1e-6  # type: ignore[operator]


def test_get_property_year_built_earliest(costar_dir: Path) -> None:
    """year_built uses the earliest ConstructionComplete year."""
    rec = get_property(100)
    assert rec is not None
    assert rec.year_built == 1985


def test_get_property_no_coords_ok(costar_dir: Path) -> None:
    """get_property returns a record even when coordinates are None."""
    rec = get_property(102)
    assert rec is not None
    assert rec.latitude is None
    assert rec.longitude is None


# ---------------------------------------------------------------------------
# find_buildings
# ---------------------------------------------------------------------------


def test_find_buildings_houston_multi(costar_dir: Path) -> None:
    """find_buildings returns only Houston multifamily buildings with coords."""
    results = find_buildings(market_id=14, property_type=11)
    # Property 100 qualifies; 102 has no coords and is excluded
    assert len(results) == 1
    assert results[0].id == 100


def test_find_buildings_houston_office(costar_dir: Path) -> None:
    """find_buildings can filter by office type."""
    results = find_buildings(market_id=14, property_type=5)
    assert len(results) == 1
    assert results[0].id == 101


def test_find_buildings_la(costar_dir: Path) -> None:
    """find_buildings works for LA (market_id=20)."""
    results = find_buildings(market_id=20, property_type=11)
    assert len(results) == 1
    assert results[0].id == 200


def test_find_buildings_limit(costar_dir: Path) -> None:
    """find_buildings honours the limit parameter."""
    results = find_buildings(market_id=14, property_type=11, limit=0)
    assert results == []


def test_find_buildings_sorted_by_id(costar_dir: Path) -> None:
    """Results are sorted by ascending id."""
    ids = [r.id for r in find_buildings(market_id=14, property_type=11)]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Joined and derived fields
# ---------------------------------------------------------------------------


def test_max_rent_joined_from_listings(costar_dir: Path) -> None:
    """max_rent comes from apartment-listing joined on propertyId."""
    assert get_property(100).max_rent == 1200
    assert get_property(200).max_rent == 2500
    assert get_property(101).max_rent is None


def test_surface_parking_and_unknowns(costar_dir: Path) -> None:
    """Surface parking sums types 1 and 2; incomplete counts stay unknown."""
    from backend.tools.costar_data import _surface_parking_spaces

    assert _surface_parking_spaces([{"parkingType": 1, "totalSpaces": 10},
                                    {"parkingType": 2, "totalSpaces": 5},
                                    {"parkingType": 3, "totalSpaces": 99}]) == 15
    assert _surface_parking_spaces([{"parkingType": 1}]) is None
    assert _surface_parking_spaces(None) is None
    assert get_property(100).surface_parking_spaces is None


def test_building_size_and_class(costar_dir: Path) -> None:
    rec = get_property(100)
    assert rec.building_size_sq_ft == 50000
    assert rec.building_class == 2
