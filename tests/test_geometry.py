"""Pure-math geometry tests: no network, no model, just CoStar field arithmetic."""

from backend.schemas import Building
from backend.tools import geometry


def test_roof_footprint_prefers_gross_typical_floor() -> None:
    b = Building(id=1, gross_typical_floor_sq_ft=5000, floor_areas_sq_ft=[9999],
                 building_size_sq_ft=99999, num_stories_above_grade=1)
    assert geometry.roof_footprint_sq_ft(b) == 5000


def test_roof_footprint_falls_back_to_largest_floor_area() -> None:
    b = Building(id=2, floor_areas_sq_ft=[3000, 4500, 4000], building_size_sq_ft=99999, num_stories_above_grade=1)
    assert geometry.roof_footprint_sq_ft(b) == 4500


def test_roof_footprint_falls_back_to_gba_over_stories() -> None:
    b = Building(id=3, building_size_sq_ft=40000, num_stories_above_grade=4)
    assert geometry.roof_footprint_sq_ft(b) == 10000


def test_roof_footprint_none_when_everything_missing() -> None:
    assert geometry.roof_footprint_sq_ft(Building(id=4)) is None


def test_roof_footprint_none_when_stories_missing_for_gba_fallback() -> None:
    b = Building(id=5, building_size_sq_ft=40000)
    assert geometry.roof_footprint_sq_ft(b) is None


def test_building_height_prefers_costar_measured_value() -> None:
    b = Building(id=6, building_height_ft=186.0, num_stories_above_grade=20)
    assert geometry.building_height_ft(b) == 186.0


def test_building_height_falls_back_to_stories_times_assumption() -> None:
    b = Building(id=7, num_stories_above_grade=3)
    assert geometry.building_height_ft(b) == 3 * geometry.FLOOR_TO_FLOOR_FT


def test_building_height_none_when_missing() -> None:
    assert geometry.building_height_ft(Building(id=8)) is None


def test_lot_size_converts_acres_to_sq_ft() -> None:
    b = Building(id=9, lot_size=1.0, lot_size_measurement_type=2)
    assert geometry.lot_size_sq_ft(b) == geometry.SQ_FT_PER_ACRE


def test_lot_size_already_sq_ft_is_passed_through() -> None:
    b = Building(id=10, lot_size=5000.0, lot_size_measurement_type=1)
    assert geometry.lot_size_sq_ft(b) == 5000.0


def test_lot_size_none_when_missing() -> None:
    assert geometry.lot_size_sq_ft(Building(id=11)) is None


def test_open_ground_is_lot_minus_footprint() -> None:
    b = Building(id=12, lot_size=2.0, lot_size_measurement_type=2,
                 gross_typical_floor_sq_ft=20000)
    expected = 2 * geometry.SQ_FT_PER_ACRE - 20000
    assert geometry.open_ground_sq_ft(b) == expected


def test_open_ground_none_when_footprint_unknown() -> None:
    b = Building(id=13, lot_size=2.0, lot_size_measurement_type=2)
    assert geometry.open_ground_sq_ft(b) is None
