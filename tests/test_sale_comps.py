"""Sale comps joined onto buildings: latest priced sale wins; portfolio totals are never attributed."""

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.tools import costar_data
from scripts.prepare_showcase import latest_sales

SALES = [
    {"id": 1, "soldDate": "2015-03-01T12:00:00-05:00", "soldForPrice": 8_000_000, "salePriceStatus": 3,
     "pricePerUnit": 80_000, "entities": [{"propertyId": 5}]},
    {"id": 2, "soldDate": "2021-06-15T12:00:00-04:00", "soldForPrice": 12_000_000, "salePriceStatus": 5,
     "pricePerUnit": 120_000, "entities": [{"propertyId": 5}]},
    # A portfolio sale: property 5 has no allocated price (ignored); property 6 does.
    {"id": 3, "soldDate": "2023-01-01T12:00:00-05:00", "soldForPrice": 90_000_000, "salePriceStatus": 3,
     "pricePerUnit": 50_000, "entities": [{"propertyId": 5}, {"propertyId": 6, "allocatedPrice": 30_000_000}]},
    # An unpriced sale is ignored entirely.
    {"id": 4, "soldDate": "2024-01-01T12:00:00-05:00", "soldForPrice": None, "salePriceStatus": 0,
     "entities": [{"propertyId": 5}]},
]
EXPECTED_5 = {"last_sale_price": 12_000_000.0, "last_sale_price_status": 5,
              "last_sale_price_per_unit": 120_000.0, "last_sale_date": "2021-06-15"}
EXPECTED_6 = {"last_sale_price": 30_000_000.0, "last_sale_price_status": 3,
              "last_sale_price_per_unit": None, "last_sale_date": "2023-01-01"}


@pytest.fixture
def kit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "costar" / "data"
    data.mkdir(parents=True)
    (data / "commercial-property.jsonl").write_text(json.dumps({
        "id": 5, "propertyType": 11,
        "primaryAddress": {"researchMarketId": 14, "latitude": 29.7, "longitude": -95.4},
        "propertyApartment": {"totalNumUnits": 100},
    }) + "\n", encoding="utf-8")
    (data / "commercial-sale-comp.jsonl").write_text("".join(json.dumps(s) + "\n" for s in SALES), encoding="utf-8")
    monkeypatch.setattr(costar_data.config, "COSTAR_DATA_DIR", data)
    return data


def test_sale_map_and_building_fields(kit: Path) -> None:
    assert costar_data._sale_map() == {5: EXPECTED_5, 6: EXPECTED_6}
    building = costar_data.get_property(5)
    assert building.last_sale_price == 12_000_000 and building.last_sale_date == "2021-06-15"
    assert building.last_sale_price_status == 5 and building.last_sale_price_per_unit == 120_000


def test_sale_map_is_empty_without_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(costar_data.config, "COSTAR_DATA_DIR", tmp_path)
    assert costar_data._sale_map() == {}


def test_prepare_showcase_latest_sales_matches_the_loader() -> None:
    frame = pd.DataFrame(SALES)
    assert latest_sales(frame) == {5: EXPECTED_5, 6: EXPECTED_6}
    assert latest_sales(None) == {}
