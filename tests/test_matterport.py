"""Tests for backend.tools.matterport's pure summarize/height helpers (no network)."""

from backend.schemas import MatterportModel
from backend.tools import matterport


def _room(estimate_height=None, dimension_height=None, label=None, estimate_area=None):
    return {
        "label": label, "tags": [],
        "dimensions": {"height": dimension_height, "areaFloor": None, "units": "imperial"},
        "dimensionEstimates": {"height": estimate_height, "area": estimate_area, "lowerElevation": -0.1,
                               "units": "imperial"},
    }


def test_summarize_prefers_estimates_and_drops_unresolved_heights() -> None:
    """dimensionEstimates is read first; a 0.0 height (unresolved ceiling) is not a measurement."""
    model = {
        "name": "Test", "created": "2026-01-01",
        "floors": [{"id": "f1", "dimensions": {"areaFloor": 11650.0, "units": "imperial"},
                    "dimensionEstimates": {"area": 9681.6}}],
        "rooms": [_room(estimate_height=9.0, label="Breakroom"), _room(estimate_height=9.2),
                  _room(estimate_height=0.0, dimension_height=8.8, label="Warehouse"),
                  _room(estimate_height=0.0, dimension_height=0.0)],
    }
    summary = matterport.summarize(model)
    assert summary["floor_count"] == 1 and summary["scanned_floor_area"] == 11650.0
    assert summary["room_count"] == 4 and summary["labelled_rooms"] == ["Breakroom", "Warehouse"]
    assert summary["measured_ceiling_heights"] == [9.0, 9.2, 8.8]  # falls back to dimensions.height
    assert summary["units"] == "imperial"
    assert summary["rooms"][0]["height"] == 9.0 and summary["rooms"][0]["lower_elevation"] == -0.1


def test_building_height_is_median_of_measured_ceiling_heights() -> None:
    model = MatterportModel(id="m-1", metadata={"measured_ceiling_heights": [9.0, 9.1, 9.3]})
    assert matterport.building_height_ft(model) == 9.1


def test_building_height_ignores_near_zero_placeholders() -> None:
    model = MatterportModel(id="m-2", metadata={"measured_ceiling_heights": [0.0, 0.3, 9.0]})
    assert matterport.building_height_ft(model) == 9.0


def test_building_height_none_without_measured_rooms() -> None:
    assert matterport.building_height_ft(MatterportModel(id="m-3", metadata={"measured_ceiling_heights": []})) is None
    assert matterport.building_height_ft(MatterportModel(id="m-4")) is None


def test_model_id_for_reads_the_kit_csv(tmp_path, monkeypatch) -> None:
    kit = tmp_path / "costar"
    (kit / "data").mkdir(parents=True)
    (kit / "commercial-properties.csv").write_text(
        "PropertyId,MatterportModelId,Address,City,State\n23296,oNAvrNiXZS7,x,Houston,TX\n99,,none,Houston,TX\n")
    monkeypatch.setattr(matterport.config, "COSTAR_DATA_DIR", kit / "data")
    matterport._model_ids.cache_clear()
    assert matterport.model_id_for(23296) == "oNAvrNiXZS7"
    assert matterport.model_id_for(99) is None
    matterport._model_ids.cache_clear()
