"""Test exact question completeness rules and parking count denominators."""

from copy import deepcopy
from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.check_questions import count_options, generate_report, render_report


def apartment(property_id: int, market: int = 14) -> dict[str, object]:
    """Create complete controls with reported zero vacancy and units."""
    return {
        "id": property_id, "propertyType": 11, "buildingClass": 2,
        "primaryAddress": {"researchMarketId": market, "latitude": 30, "longitude": -95},
        "propertyApartment": {"totalNumUnits": 0, "vacancyPercent": 0},
        "constructionEvents": [{"eventType": 2, "year": 1980}, {"eventType": 2, "year": 1981}],
        "parking": [{"parkingType": 1, "totalSpaces": 10}, {"parkingType": 1, "totalSpaces": 5}],
    }


def test_property_join_intersections_and_zero_values() -> None:
    """Listing ids do not identify properties; NaN != 0 cannot manufacture rent."""
    records = [apartment(1), apartment(2), apartment(3), apartment(4, 20), apartment(5, 20)]
    records[1]["propertyApartment"]["totalNumUnits"] = 100
    records[2]["propertyApartment"]["vacancyPercent"] = None
    records[4]["propertyApartment"]["vacancyPercent"] = None
    listings = pd.DataFrame([
        {"id": 5, "propertyId": 1, "maxRent": 2000},
        {"id": 1, "propertyId": 2, "maxRent": 0},
        {"id": 2, "propertyId": 3, "maxRent": float("nan")},
        {"id": 4, "propertyId": 5, "maxRent": 3000},
    ])
    result = count_options(pd.DataFrame(records), listings)
    houston, la = result["cities"][14], result["cities"][20]
    assert houston["controls_complete"] == 3
    assert (houston["with_rent"], houston["with_vacancy"], houston["with_both"]) == (1, 2, 1)
    assert houston["zero_units"] == 2
    assert houston["positive_units_controls"] == 1
    assert (la["with_rent"], la["with_vacancy"], la["with_both"]) == (1, 1, 0)
    assert result["parking_entries"][14]["1"] == 6
    assert result["parking_properties"][14]["1"] == 3


@pytest.mark.parametrize("missing", ["latitude", "longitude", "units", "class_null", "class_zero", "year_null", "renovation"])
def test_every_required_field_gates_the_cohort(missing: str) -> None:
    """Missing controls cannot pass via nonzero comparisons or renovation years."""
    record = apartment(1)
    if missing in ("latitude", "longitude"):
        record["primaryAddress"][missing] = None
    elif missing == "units":
        record["propertyApartment"]["totalNumUnits"] = None
    elif missing.startswith("class"):
        record["buildingClass"] = 0 if missing == "class_zero" else float("nan")
    else:
        record["constructionEvents"] = [{"eventType": 4 if missing == "renovation" else 2, "year": 2020 if missing == "renovation" else None}]
    listings = pd.DataFrame([{"propertyId": 1, "maxRent": 2000}])
    city = count_options(pd.DataFrame([record]), listings)["cities"][14]
    assert city["multifamily"] == 1
    assert city["controls_complete"] == city["with_rent"] == city["with_vacancy"] == city["with_both"] == 0


def test_scope_and_parking_types_before_completeness_filtering() -> None:
    """Count incomplete properties' parking but exclude other markets/types."""
    incomplete = apartment(1)
    incomplete["buildingClass"] = 0
    incomplete["parking"] = [{"parkingType": 0}, {"parkingType": 2}, {"parkingType": None}]
    office = deepcopy(incomplete)
    office.update(id=2, propertyType=5)
    other_market = apartment(3, 99)
    la = apartment(4, 20)
    la["parking"] = None
    result = count_options(pd.DataFrame([incomplete, office, other_market, la]), pd.DataFrame([{"propertyId": 1, "maxRent": 2000}]))
    assert result["cities"][14]["multifamily"] == 1
    assert result["cities"][20]["multifamily"] == 1
    assert result["cities"][14]["controls_complete"] == 0
    assert dict(result["parking_entries"][14]) == {"0": 1, "2": 1, "missing": 1}
    assert result["cities"][20]["without_parking_entries"] == 1
    report = render_report(result, {"0": "Unknown", "2": "SurfaceTandem"}, False)
    assert "| 2 | SurfaceTandem | 1 | 1 | 0 | 0 |" in report
    assert "| missing | Missing | 1 | 1 | 0 | 0 |" in report


@pytest.mark.parametrize("duplicate", ["property", "listing"])
def test_duplicate_keys_are_rejected(duplicate: str) -> None:
    """Duplicate rows cannot silently inflate counts or choose a rent outcome."""
    properties = pd.DataFrame([apartment(1)] * (2 if duplicate == "property" else 1))
    listings = pd.DataFrame([{"propertyId": 1, "maxRent": 2000}] * (2 if duplicate == "listing" else 1))
    with pytest.raises(ValueError, match="unique"):
        count_options(properties, listings)


def test_local_json_lines_to_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate the report from local files even when cwd is outside the repo."""
    data_dir = tmp_path / "costar" / "data"
    data_dir.mkdir(parents=True)
    schemas = data_dir.parent / "schemas"
    schemas.mkdir()
    (schemas / "enums.json").write_text(json.dumps({"ParkingType": {"1": "Surface"}}))
    pd.DataFrame([apartment(1)]).to_json(data_dir / "commercial-property.jsonl", orient="records", lines=True)
    pd.DataFrame([{"propertyId": 1, "maxRent": 2000}]).to_json(data_dir / "apartment-listing.jsonl", orient="records", lines=True)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "notes" / "question_options.md"
    assert generate_report(data_dir, output) == output
    report = output.read_text()
    assert "| Houston (14) | 1 | 1 | 1 | 1 | 1 |" in report
    assert "| Los Angeles (20) | 0 | 0 | 0 | 0 | 0 |" in report
    assert "| 1 | Surface | 2 | 1 | 0 | 0 |" in report
