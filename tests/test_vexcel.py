"""Pure-math tests for backend.tools.vexcel's parcel_bearing_deg helper."""

from backend.schemas import VexcelPropertyInfo
from backend.tools import vexcel


def _info(geometry: dict | None, lat: float = 0.0, lon: float = 0.0) -> VexcelPropertyInfo:
    features = [] if geometry is None else [{
        "geometry": geometry,
        "properties": {},
    }]
    return VexcelPropertyInfo(
        id="v-1", latitude=lat, longitude=lon, attributes={},
        raw_response={"features": features},
    )


def test_bearing_of_east_west_rectangle_is_near_90_or_0() -> None:
    # A rectangle whose longest edges run east-west (lon varies, lat barely does).
    rectangle = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, 0.0], [0.01, 0.0], [0.01, 0.0001], [0.0, 0.0001], [0.0, 0.0],
        ]],
    }
    bearing = vexcel.parcel_bearing_deg(_info(rectangle, lat=0.00005, lon=0.005))
    assert bearing is not None
    # East-west edge bearing is ~90 (mod 180, so also valid near 90).
    assert abs(bearing - 90) < 5


def test_bearing_of_north_south_rectangle_is_near_0_or_180() -> None:
    rectangle = {
        "type": "Polygon",
        "coordinates": [[
            [0.0, 0.0], [0.0001, 0.0], [0.0001, 0.01], [0.0, 0.01], [0.0, 0.0],
        ]],
    }
    bearing = vexcel.parcel_bearing_deg(_info(rectangle, lat=0.005, lon=0.00005))
    assert bearing is not None
    assert bearing < 5 or bearing > 175


def test_bearing_is_none_when_no_feature_geometry() -> None:
    assert vexcel.parcel_bearing_deg(_info(None)) is None


def test_bearing_is_none_for_point_geometry() -> None:
    point = {"type": "Point", "coordinates": [0.0, 0.0]}
    assert vexcel.parcel_bearing_deg(_info(point)) is None


def test_bearing_uses_largest_polygon_in_a_multipolygon() -> None:
    # Real Vexcel parcels come back as MultiPolygon; use the largest piece.
    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[5.0, 5.0], [5.0001, 5.0], [5.0001, 5.0001], [5.0, 5.0001], [5.0, 5.0]]],  # tiny sliver
            [[[0.0, 0.0], [0.01, 0.0], [0.01, 0.0001], [0.0, 0.0001], [0.0, 0.0]]],  # main parcel, east-west
        ],
    }
    bearing = vexcel.parcel_bearing_deg(_info(multi, lat=0.00005, lon=0.005))
    assert bearing is not None
    assert abs(bearing - 90) < 5
