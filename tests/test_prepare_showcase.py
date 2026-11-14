"""Verify showcase cohort eligibility, reproducibility and parking references."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_showcase import (
    build_samples,
    completion_year,
    prepare_candidates,
    surface_parking_spaces,
)


def apartment(property_id: int, market_id: int) -> dict[str, object]:
    """Build a synthetic property with known controls and reported zero vacancy."""
    return {
        "id": property_id,
        "propertyType": 11,
        "primaryAddress": {"researchMarketId": market_id, "latitude": 30, "longitude": -95},
        "buildingClass": 2,
        "constructionEvents": [{"eventType": 2, "year": 1980}],
        "propertyApartment": {"totalNumUnits": 100, "vacancyPercent": 0},
        "parking": [{"parkingType": 1, "totalSpaces": 20}],
    }


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    """Write complete and incomplete communities to a small local JSON Lines kit."""
    data_dir = tmp_path / "costar" / "data"
    data_dir.mkdir(parents=True)
    properties = [apartment(i, 14) for i in range(1, 6)]
    properties[3]["constructionEvents"] = [{"eventType": 4, "year": 2020}]
    properties[4]["buildingClass"] = 0
    la = [apartment(i, 20) for i in (100, 101, 102)]
    for record in la:
        record["propertyApartment"]["vacancyPercent"] = None
    properties.extend(la)
    listings = [
        {"id": 10 + i, "propertyId": i, "maxRent": 2000 if i != 3 else 0}
        for i in range(1, 6)
    ]
    for name, records in (("commercial-property", properties), ("apartment-listing", listings)):
        pd.DataFrame(records).to_json(data_dir / f"{name}.jsonl", orient="records", lines=True)
    return data_dir


def test_city_specific_eligibility_and_known_zero(kit: Path, tmp_path: Path) -> None:
    """Houston needs both outcomes; LA can have neither; vacancy zero is valid."""
    output_dir = tmp_path / "samples"
    manifest = build_samples(kit, output_dir, size=2)
    houston = pd.read_csv(output_dir / "houston.csv")
    los_angeles = pd.read_csv(output_dir / "los_angeles.csv")
    assert set(houston["property_id"]) == {1, 2}
    assert houston["max_rent"].gt(0).all()
    assert houston["vacancy_percent"].eq(0).all()
    assert los_angeles["market_id"].eq(20).all()
    assert los_angeles["max_rent"].isna().all()
    assert los_angeles["vacancy_percent"].isna().all()
    assert manifest["cities"]["houston"]["eligible_count"] == 2
    assert manifest["cities"]["los_angeles"]["eligible_count"] == 3
    assert manifest["cities"]["los_angeles"]["required_outcomes"] == []
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest


def test_sampling_is_reproducible_when_inputs_reorder(kit: Path, tmp_path: Path) -> None:
    """The same seed and property set produce byte-identical sample artifacts."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_samples(kit, first, size=2, seed=17)
    for path in kit.glob("*.jsonl"):
        lines = path.read_text().splitlines()
        path.write_text("\n".join(reversed(lines)) + "\n")
    build_samples(kit, second, size=2, seed=17)
    for filename in ("houston.csv", "los_angeles.csv", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_short_cohort_does_not_write_partial_samples(kit: Path, tmp_path: Path) -> None:
    """Insufficient eligibility fails before creating either city's sample."""
    output_dir = tmp_path / "samples"
    with pytest.raises(ValueError, match="houston: need 50 eligible properties, found 2"):
        build_samples(kit, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("parking", "expected"),
    [
        (None, None), ([], None),
        ([{"parkingType": 3, "totalSpaces": 100}], None),
        ([{"parkingType": 1, "totalSpaces": 0}], 0),
        ([{"parkingType": 1, "totalSpaces": 10}, {"parkingType": 2, "totalSpaces": 5},
          {"parkingType": 3, "totalSpaces": 100}, {"parkingType": 8, "totalSpaces": 30}], 15),
        ([{"parkingType": 1, "totalSpaces": 10}, {"parkingType": 2, "totalSpaces": None}], None),
    ],
)
def test_surface_parking_reference(parking: object, expected: int | None) -> None:
    """Exclude covered/ambiguous types and preserve unknown versus zero spaces."""
    assert surface_parking_spaces(parking) == expected


@pytest.mark.parametrize("missing", ["latitude", "units", "year", "class", "vacancy"])
def test_incomplete_houston_rows_are_ineligible(
    kit: Path, tmp_path: Path, missing: str
) -> None:
    """Each required coordinate, control and outcome actually gates Houston."""
    path = kit / "commercial-property.jsonl"
    records = pd.read_json(path, lines=True).to_dict("records")
    record = records[0]
    if missing == "latitude":
        record["primaryAddress"]["latitude"] = None
    elif missing == "units":
        record["propertyApartment"]["totalNumUnits"] = None
    elif missing == "year":
        record["constructionEvents"] = [{"eventType": 4, "year": 2020}]
    elif missing == "class":
        record["buildingClass"] = 0
    else:
        record["propertyApartment"]["vacancyPercent"] = None
    pd.DataFrame(records).to_json(path, orient="records", lines=True)
    with pytest.raises(ValueError, match="houston: need 2 eligible properties, found 1"):
        build_samples(kit, tmp_path / "samples", size=2)


def test_duplicate_listing_join_is_rejected() -> None:
    """Two listing rows cannot silently choose an outcome for the same property."""
    listings = pd.DataFrame([
        {"id": 10, "propertyId": 1, "maxRent": 2000},
        {"id": 11, "propertyId": 1, "maxRent": 3000},
    ])
    with pytest.raises(ValueError, match="unique propertyId"):
        prepare_candidates(pd.DataFrame([apartment(1, 14)]), listings)


def test_construction_year_excludes_renovations() -> None:
    """Completion events determine year built, not the most recent renovation."""
    assert completion_year([
        {"eventType": 2, "year": 1985}, {"eventType": 4, "year": 2020},
        {"eventType": 2, "year": 1980},
    ]) == 1980
    assert completion_year([{"eventType": 4, "year": 2020}]) is None


def test_showcase_definitions_reference_available_sample_fields(
    kit: Path, tmp_path: Path
) -> None:
    """Every question can locate its cohort, controls, outcome and reference fields."""
    path = Path(__file__).resolve().parents[1] / "data" / "showcase" / "questions.json"
    definitions = json.loads(path.read_text())
    output_dir = tmp_path / "samples"
    manifest = build_samples(kit, output_dir, size=2)
    for control in definitions["controls"]:
        assert control["name"] in manifest["field_sources"]
    for question in definitions["questions"]:
        sample = pd.read_csv(output_dir / Path(question["sample"]).name)
        assert sample["market_id"].eq(question["market_id"]).all()
        assert sample["property_type"].eq(question["property_type"]).all()
        assert set(question.get("controls", [])) <= set(sample.columns)
        if "outcome" in question:
            assert sample[question["outcome"]["name"]].notna().all()
        if "validation" in question:
            assert question["validation"]["reference"] in sample.columns
