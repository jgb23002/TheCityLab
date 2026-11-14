"""Verify outcome arithmetic, source attribution and city-specific rankings."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.pick_questions import (
    annual_rent_per_sq_ft,
    apartment_rents,
    commercial_rent_properties,
    generate_report,
    nonzero_mean,
    rank_options,
    render_report,
)


def property_record(property_id: int, market: int, property_type: int) -> dict[str, object]:
    """Create a synthetic property with only the requested coordinates/controls."""
    return {
        "id": property_id, "propertyType": property_type,
        "primaryAddress": {"researchMarketId": market, "latitude": 30, "longitude": -95},
        "grossBuildingAreaSquareFeet": 10000,
        "constructionEvents": [{"eventType": 2, "year": 1980}],
        # These former controls must not filter the new comparison.
        "buildingClass": 0, "propertyApartment": {"totalNumUnits": 0, "vacancyPercent": 10},
    }


def empty_commercial() -> pd.DataFrame:
    """Return a source with no commercial asking quotes."""
    return pd.DataFrame(columns=["id", "propertyId", "forLeaseSummary"])


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [((1000, 3000), 2000), ((0, 2000), 2000), ((1500, None), 1500),
     ((float("nan"), 2500), 2500), ((0, None), None), ((0, 0), None)],
)
def test_apartment_rent_ignores_missing_and_zero_bounds(bounds: tuple, expected: float | None) -> None:
    """A single recorded bound remains usable without being halved by zero."""
    assert nonzero_mean(*bounds) == expected


def test_apartment_rent_joins_on_property_id() -> None:
    """Listing ids are not building ids and missing foreign keys do not link."""
    frame = pd.DataFrame([
        {"id": 100, "propertyId": 1, "minRent": 0, "maxRent": 2000},
        {"id": 1, "propertyId": None, "minRent": 1000, "maxRent": 3000},
    ])
    assert apartment_rents(frame) == {1: 2000}
    with pytest.raises(ValueError, match="unique propertyId"):
        apartment_rents(pd.concat([frame, frame.iloc[[0]]]))


@pytest.mark.parametrize(
    ("basis", "measurement", "expected"),
    [(1, 1, 24), (2, 1, 2), (0, 1, None), (2, 0, None), (2, 2, None), (2, 3, None)],
)
def test_commercial_rate_requires_known_units(basis: int, measurement: int, expected: float | None) -> None:
    """Monthly rates annualize; acre, meter and unspecified rates cannot pose as per-SF."""
    assert annual_rent_per_sq_ft(0, 2, basis, measurement) == expected


def test_group_quotes_follow_spaces_not_the_group_header() -> None:
    """A group-wide rollup cannot create a rate for an unquoted property."""
    frame = pd.DataFrame([
        {
            "id": 100, "propertyId": 1, "listingType": 16,
            "forLeaseSummary": {
                "askingRateLow": 20, "askingRateHigh": 30, "askingRateBasis": 2,
                "askingRateMeasurementType": 1,
                "spaces": [
                    {"propertyId": 2, "rateLow": 10, "rateHigh": 20, "rateBasis": 2},
                    {"propertyId": 2, "rateLow": 10, "rateHigh": 20, "rateBasis": 2},
                    {"propertyId": 3, "rateLow": 0, "rateHigh": 0, "rateBasis": 2},
                    {"propertyId": None, "rateLow": 10, "rateHigh": 20, "rateBasis": 2},
                ],
            },
        },
        {
            "id": 101, "propertyId": 4, "listingType": 14, "listingStatus": 5,
            "forLeaseSummary": {
                "askingRateLow": 0, "askingRateHigh": 0, "askingRateBasis": 2,
                "askingRateMeasurementType": 1,
                "spaces": [{"propertyId": None, "rateLow": 10, "rateHigh": 20, "rateBasis": 1,
                            "dateOffMarket": "2020-01-01", "isLeasePending": True}],
            },
        },
    ])
    # Historical quotes qualify for coverage; no current-market estimate is made.
    assert commercial_rent_properties(frame) == {2, 4}


def test_top_three_per_city_before_filtering_and_descending_sort() -> None:
    """Population selects types, controls gate outcomes, and ties use numeric code."""
    records = []
    next_id = 1
    for market, type_counts in ((14, [(11, 4), (5, 3), (2, 2), (6, 2)]),
                                (20, [(6, 4), (5, 3), (7, 2), (11, 1)])):
        for code, count in type_counts:
            for _ in range(count):
                records.append(property_record(next_id, market, code))
                next_id += 1
    # Most common Houston type remains selected despite having no valid size.
    for record in records[:4]:
        record["grossBuildingAreaSquareFeet"] = None
    apartments = pd.DataFrame([{"propertyId": 1, "minRent": 1000, "maxRent": 2000}])
    options, selected = rank_options(pd.DataFrame(records), apartments, empty_commercial())
    assert [r["type"] for r in selected if r["market"] == 14] == [11, 5, 2]
    assert [r["type"] for r in selected if r["market"] == 20] == [6, 5, 7]
    assert [r["count"] for r in options] == sorted((r["count"] for r in options), reverse=True)
    houston_mf = [r for r in options if r["city"] == "Houston" and r["type"] == 11]
    assert len(houston_mf) == 3 and all(r["count"] == 0 for r in houston_mf)
    report = render_report(options, selected, {"2": "Industrial", "6": "GeneralRetail"})
    assert "GeneralRetail (6) ties at 2" in report
    assert "## Occupancy/vacancy outcome: unsupported" in report
    assert all("vacancy" not in row["outcome"].lower() for row in options)


@pytest.mark.parametrize("missing", ["latitude", "longitude", "size", "year"])
def test_all_coordinates_and_controls_are_required(missing: str) -> None:
    """Neither renovation years nor an available rent can rescue missing controls."""
    record = property_record(1, 14, 11)
    if missing in ("latitude", "longitude"):
        record["primaryAddress"][missing] = None
    elif missing == "size":
        record["grossBuildingAreaSquareFeet"] = None
    else:
        record["constructionEvents"] = [{"eventType": 4, "year": 2020}]
    apartments = pd.DataFrame([{"propertyId": 1, "minRent": 1000, "maxRent": 2000}])
    options, _ = rank_options(pd.DataFrame([record]), apartments, empty_commercial())
    assert all(row["count"] == 0 for row in options)


def test_generate_from_local_files_without_a_tenancy_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the requested report with correct joins and no tenancy-rate fabrication."""
    data_dir = tmp_path / "costar" / "data"
    data_dir.mkdir(parents=True)
    datasets = {
        "commercial-property": pd.DataFrame([property_record(1, 14, 11), property_record(2, 20, 5)]),
        "apartment-listing": pd.DataFrame([{"propertyId": 1, "minRent": 0, "maxRent": 2000}]),
        "commercial-listing-summary": pd.DataFrame([
            {"id": 3, "propertyId": 2, "forLeaseSummary": {
                "askingRateLow": 20, "askingRateHigh": 30, "askingRateBasis": 2,
                "askingRateMeasurementType": 1,
            }},
        ]),
    }
    for name, frame in datasets.items():
        frame.to_json(data_dir / f"{name}.jsonl", orient="records", lines=True)
    schemas = data_dir.parent / "schemas"
    schemas.mkdir()
    (schemas / "enums.json").write_text(json.dumps({"PropertyType": {"11": "Multifamily", "5": "Office"}}))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "notes" / "question_options.md"
    assert generate_report(data_dir, output) == output
    text = output.read_text()
    assert "| Houston | Multifamily (11) | Apartment asking rent (monthly midpoint) | 1 |" in text
    assert "| Los Angeles | Office (5) | Commercial asking rent per SF (annualized) | 1 |" in text
    assert "not substituted" in text
