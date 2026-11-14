"""Tests for backend.tools.matterport's pure building_height_ft aggregation."""

from backend.schemas import MatterportModel
from backend.tools import matterport


def _model(rooms: list[dict]) -> MatterportModel:
    return MatterportModel(id="m-1", metadata={"rooms": rooms, "floors": []})


def test_building_height_is_median_of_positive_room_heights() -> None:
    model = _model([
        {"dimensionEstimates": {"height": 9.0}},
        {"dimensionEstimates": {"height": 9.1}},
        {"dimensionEstimates": {"height": 9.3}},
    ])
    assert matterport.building_height_ft(model) == 9.1


def test_building_height_excludes_zero_height_rooms() -> None:
    """A room whose ceiling plane couldn't be resolved comes back 0.0, not a real height."""
    model = _model([
        {"dimensionEstimates": {"height": 0.0}},
        {"dimensionEstimates": {"height": 9.0}},
    ])
    assert matterport.building_height_ft(model) == 9.0


def test_building_height_none_when_no_rooms_have_height() -> None:
    model = _model([{"dimensionEstimates": {"height": 0.0}}])
    assert matterport.building_height_ft(model) is None


def test_building_height_none_when_no_rooms() -> None:
    assert matterport.building_height_ft(_model([])) is None
