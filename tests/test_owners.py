"""Owner lookup tests on a tiny fake CoStar kit."""

import json

import pytest

from backend.tools import owners


@pytest.fixture
def kit(tmp_path, monkeypatch: pytest.MonkeyPatch):
    d = tmp_path / "data"
    d.mkdir()
    rows = {
        "company-detail.jsonl": [{"id": 500, "name": "Acme HQ", "profile": {"companyName": "Acme Holdings"}}],
        "commercial-property.jsonl": [
            {"id": 1, "buildingName": "Tower", "primaryAddress": {"streetNumber": "1", "streetName": "Main", "streetType": "St", "city": "Houston", "state": "TX"},
             "contacts": [{"contactRole": 3, "locationId": 999}, {"contactRole": 7, "locationId": 500}]},
            {"id": 2, "buildingName": None, "primaryAddress": None, "contacts": None},
            {"id": 3, "buildingName": "Lofts", "primaryAddress": None, "contacts": []},
        ],
        "commercial-sale-comp.jsonl": [
            {"soldDate": "2010-01-01T00:00:00", "entities": [{"propertyId": 3}], "contacts": [{"contactRole": 43, "locationName": "Old Buyer LLC"}]},
            {"soldDate": "2020-05-01T00:00:00", "entities": [{"propertyId": 3}], "contacts": [{"contactRole": 42, "locationName": "New Buyer LLC"}]},
        ],
    }
    for name, recs in rows.items():
        (d / name).write_text("\n".join(json.dumps(r) for r in recs))
    monkeypatch.setattr(owners.config, "COSTAR_DATA_DIR", d)
    monkeypatch.setattr(owners, "INDEX_PATH", tmp_path / "cache" / "owner_index.json")
    owners.load_index.cache_clear()
    yield
    owners.load_index.cache_clear()


def test_owner_record_then_latest_sale_buyer(kit) -> None:
    assert owners.lookup(1) == {"building_name": "Tower", "address": "1 Main St, Houston, TX",
                                "owner": "Acme Holdings", "owner_source": "CoStar owner record"}
    three = owners.lookup(3)
    assert three["owner"] == "New Buyer LLC" and "2020-05-01" in three["owner_source"]
    assert owners.lookup(2)["owner"] is None and owners.lookup(2)["building_name"] is None
    assert owners.lookup(404)["owner"] is None


def test_index_is_cached_to_disk(kit) -> None:
    owners.load_index()
    assert owners.INDEX_PATH.exists()
