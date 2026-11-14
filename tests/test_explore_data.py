"""Exercise report counts and interpretation using a small synthetic local kit."""

import json
from pathlib import Path

import pytest

from scripts.explore_data import count_rows, generate_summary, is_nonempty


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    """Create JSON Lines with missing values, multiple list entries and scans."""
    root = tmp_path / "costar"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    datasets = {
        "commercial-property": [
            {
                "id": 1,
                "propertyType": 5,
                "primaryAddress": {"researchMarketId": 14, "latitude": 29.7, "longitude": -95.4},
                "buildingClass": 0,
                "rentableBuildingAreaSquareFeet": 1000,
                "constructionEvents": [
                    {"eventType": 3, "year": 2020},
                    {"eventType": 2, "year": 1990},
                    {"eventType": 2, "year": 1991},
                ],
                "parking": [{"totalSpaces": 0}, {"totalSpaces": 10}],
                "notes": [{"text": "Unselected private content"}],
            },
            {
                "id": 2,
                "propertyType": 11,
                "primaryAddress": {"researchMarketId": 20, "latitude": 34.0, "longitude": None},
                "buildingClass": 2,
                "propertyApartment": {"totalNumUnits": 20, "vacancyPercent": 0},
                "constructionEvents": [{"eventType": 4, "year": 2021}],
                "parking": [],
            },
        ],
        "apartment-listing": [
            {
                "id": 3, "propertyId": 2, "minRent": 0, "maxRent": 1500,
                "apartmentModels": [
                    {"minRent": 0, "maxRent": 0, "rentPricingType": 1},
                    {"minRent": 1000, "maxRent": None, "rentPricingType": 2},
                    {"minRent": None, "maxRent": 2000, "rentPricingType": 3},
                ],
            },
        ],
        "commercial-listing-summary": [
            {
                "id": 4, "propertyId": 1, "listingStatus": 1,
                "forLeaseSummary": {
                    "askingRateLow": 0, "askingRateHigh": 30,
                    "totalAvailableSquareFeet": 100,
                    "spaces": [{"rateLow": 20, "rateHigh": 30, "squareFeetMax": 100}],
                },
            },
        ],
        "location-occupancy": [
            {"id": 5, "propertyId": 1, "fromDate": "2020-01-01", "toDate": None},
            {"id": 6, "propertyId": 1, "fromDate": "2010-01-01", "toDate": "2019-12-31"},
        ],
        "another-dataset": [{"id": 7, "extraColumn": "also inventoried"}],
    }
    for name, records in datasets.items():
        contents = "".join(json.dumps(record) + "\n" for record in records)
        (data_dir / f"{name}.jsonl").write_text(contents, encoding="utf-8")
    (root / "schemas").mkdir()
    (root / "schemas" / "enums.json").write_text(
        json.dumps({"PropertyType": {"5": "Office", "11": "Multifamily"}}),
        encoding="utf-8",
    )
    (root / "commercial-properties.csv").write_text(
        "PropertyId,MatterportModelId,Address,City,State\n"
        "1,model-a,,,\n1,model-b,,,\n1,model-a,,,\n2,,,,\n3,   ,,,\n",
        encoding="utf-8",
    )
    return data_dir


def test_report_counts_and_examples(kit: Path, tmp_path: Path) -> None:
    """Compute market/rent counts and unique mapped properties from real files."""
    output = tmp_path / "notes" / "data_summary.md"
    assert generate_summary(kit, output) == output
    report = output.read_text(encoding="utf-8")
    assert len(report.splitlines()) < 200
    assert "| `another-dataset.jsonl` | 1 | `id`, `extraColumn` |" in report
    assert "| 5 (Office) | 1 | 1 | 0 |" in report
    assert "| 11 (Multifamily) | 1 | 0 | 1 |" in report
    assert "| Longitude | `primaryAddress.longitude` | 1 |" in report
    assert "Both coordinates: 1/2" in report
    assert "| Parking spaces by type | `parking[].totalSpaces` | 1 |" in report
    assert "where `eventType == 2` | 1 |" in report
    assert "| `minRent` | 1 | 1 | 0 | 1 |" in report
    assert "| `apartmentModels[].minRent` | 3 | 2 | 1 | 1 |" in report
    assert "| `apartmentModels[].maxRent` | 3 | 2 | 1 | 1 |" in report
    assert "| `forLeaseSummary.askingRateLow` | 1 | 1 | 0 | 1 |" in report
    assert "properties with a non-blank `MatterportModelId`: 1; without a model: 2" in report
    assert "`toDate` is null on 1/2 rows" in report
    assert "No direct, reliable per-property occupancy/vacancy rate" in report
    assert "`dateOffMarket`" in report and "`isLeasePending`" in report
    examples = [json.loads(block.split("\n```", 1)[0]) for block in report.split("```json\n")[1:]]
    assert [example["propertyType"] for example in examples] == [5, 11]
    assert examples[0]["constructionEvents"] == [
        {"eventType": 2, "year": 1990}, {"eventType": 2, "year": 1991}
    ]
    assert examples[1]["primaryAddress"]["longitude"] is None
    assert "Unselected private content" not in report


def test_report_does_not_depend_on_cwd_or_environment(
    kit: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Changing directory and private environment values cannot affect output."""
    monkeypatch.chdir(tmp_path)
    first = generate_summary(kit, tmp_path / "first.md").read_text(encoding="utf-8")
    monkeypatch.setenv("VEXCEL_API_KEY", "private-test-sentinel")
    second = generate_summary(kit, tmp_path / "second.md").read_text(encoding="utf-8")
    assert first == second
    assert "private-test-sentinel" not in second + capsys.readouterr().out


def test_missing_input_leaves_existing_report_untouched(kit: Path, tmp_path: Path) -> None:
    """A missing required dataset fails before replacing a previous summary."""
    (kit / "commercial-property.jsonl").unlink()
    output = tmp_path / "data_summary.md"
    output.write_text("Existing summary\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="commercial-property.jsonl"):
        generate_summary(kit, output)
    assert output.read_text(encoding="utf-8") == "Existing summary\n"


def test_nested_coverage_counts_properties_once() -> None:
    """Multiple spaces are one covered property; zero is present and NaN is not."""
    records = [
        {"parking": [{"totalSpaces": 0}, {"totalSpaces": 20}]},
        {"parking": [{"totalSpaces": float("nan")}]},
        {"parking": None},
    ]
    assert count_rows(records, "parking[].totalSpaces") == 1
    assert is_nonempty(0) and is_nonempty(False)
    assert not any(is_nonempty(value) for value in (None, float("nan"), "  ", [], {}))
