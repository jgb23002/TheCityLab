"""Tests for samples, stats.run_study, Vexcel wrapper/detection and
scripts/build_sample.py. No network."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import backend.config as cfg
from backend import samples, vexcel_detect
from backend.schemas import Building, Measurement, StatsResult
from backend.tools import stats, vexcel
from scripts import build_sample

# ---------------------------------------------------------------------------
# samples.sample_buildings (uses the committed data/samples CSVs)
# ---------------------------------------------------------------------------


def test_sample_buildings_houston_has_q1_fields() -> None:
    rows = samples.sample_buildings("houston", 11, outcome="max_rent", n=50)
    assert 40 <= len(rows) <= 50
    for b in rows:
        assert isinstance(b, Building)
        assert b.latitude is not None and b.longitude is not None
        assert b.max_rent is not None and b.market_id == 14
        assert b.total_num_units and b.year_built
    assert [b.id for b in rows] == sorted(b.id for b in rows)


def test_sample_buildings_aliases_and_limit() -> None:
    assert len(samples.sample_buildings("Los Angeles", n=5)) == 5
    assert samples.sample_buildings("LA", n=0) == []


def test_sample_buildings_blank_outcome_dropped() -> None:
    rows = samples.sample_buildings("los_angeles", outcome="surface_parking_spaces", n=100)
    assert rows and all(b.surface_parking_spaces is not None for b in rows)


def test_sample_buildings_bad_input() -> None:
    with pytest.raises(ValueError):
        samples.sample_buildings("paris")
    with pytest.raises(ValueError):
        samples.sample_buildings("houston", outcome="not_a_field")


# ---------------------------------------------------------------------------
# stats.run_study
# ---------------------------------------------------------------------------


def _synthetic(n: int = 60, slope: float = 10.0) -> tuple[list[Building], list[Measurement]]:
    rng = np.random.default_rng(0)
    buildings, measurements = [], []
    for i in range(n):
        green = float(rng.uniform(0, 40))
        units = int(rng.integers(50, 500))
        year = int(rng.integers(1960, 2020))
        cls = int(rng.integers(1, 4))
        rent = 1000 + slope * green + 0.2 * units + 100 * (3 - cls) + float(rng.normal(0, 20))
        buildings.append(Building(id=i + 1, total_num_units=units, year_built=year,
                                  building_class=cls, max_rent=rent))
        measurements.append(Measurement(id=f"m{i}", property_id=i + 1, green_cover_percent=green))
    return buildings, measurements


def test_run_study_recovers_slope() -> None:
    buildings, measurements = _synthetic()
    q = "In Houston, do apartment communities with more green cover have higher asking rents?"
    result = stats.run_study(q, buildings, measurements)
    assert isinstance(result, StatsResult)
    assert (result.outcome, result.predictor) == ("max_rent", "green_cover_percent")
    assert result.sample_size == 60
    low, high = result.confidence_intervals["green_cover_percent"]
    assert low < 10 < high
    assert result.p_values["green_cover_percent"] < 0.001
    assert set(result.coefficients) == set(result.p_values) == set(result.confidence_intervals)
    assert {"intercept", "total_num_units", "year_built", "building_class_A", "building_class_B"} <= set(result.coefficients)
    assert "building_class (categorical; C baseline)" in result.controls
    assert "associated" in result.interpretation and "not a cause" in result.interpretation
    StatsResult.model_validate_json(result.model_dump_json())


def test_run_study_unknown_question_stays_null() -> None:
    buildings, measurements = _synthetic(10)
    result = stats.run_study("Which buildings are tallest?", buildings, measurements)
    assert result.coefficients is None and result.sample_size == 0
    assert any("Could not tell" in c for c in result.caveats)


def test_run_study_too_few_rows() -> None:
    buildings, measurements = _synthetic(5)
    result = stats.run_study("Does green cover go with rent?", buildings, measurements)
    assert result.coefficients is None
    assert any("not run" in c for c in result.caveats)


def test_run_study_parking_falls_back_to_costar() -> None:
    buildings, _ = _synthetic(30)
    for b in buildings:
        b.surface_parking_spaces = b.total_num_units  # some numeric value
    result = stats.run_study("Does more parking mean higher vacancy?", buildings)
    assert result.predictor == "surface_parking_spaces"
    assert result.outcome == "vacancy_percent"
    assert result.coefficients is None  # vacancy unknown in this fixture


# ---------------------------------------------------------------------------
# Vexcel wrapper + detection (fake client)
# ---------------------------------------------------------------------------

LAT, LON = 29.76, -95.39


class FakeClient:
    """Stands in for the HTTP client; returns canned data."""

    def __init__(self, image_path: Path) -> None:
        self.image_bytes = image_path.read_bytes()

    def request_json(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 30) -> dict:
        assert payload["wkt"].startswith(("POINT(-95.", "POLYGON ((-95."))  # longitude first
        if endpoint == vexcel_detect.DETECT_ENDPOINT:
            # Shape from notes/vexcel/vexcel_quickstart.py step 3.
            assert payload["images"][0]["endpoint"] == "ortho/extract"
            assert payload["model-input"]["elements"][0]["label"] == "parking space"
            min_lon, min_lat, max_lon, max_lat = vexcel._square_bounds(LAT, LON, half_m=vexcel.SQUARE_HALF_M)
            w, h = max_lon - min_lon, max_lat - min_lat

            def wkt(x0: float, y0: float, x1: float, y1: float) -> str:
                a, b, c, d = min_lon + w * x0, min_lat + h * y0, min_lon + w * x1, min_lat + h * y1
                return f"POLYGON(({a} {b},{c} {b},{c} {d},{a} {d},{a} {b}))"

            return {
                "images": [{"image-id": "ortho-1"}, {"image-id": "obl-1"}],
                "features": [{"properties": {"label": "parking space", "detections": [
                    {"imageId": "ortho-1", "confidence": 0.9, "bbox": wkt(0.1, 0.1, 0.2, 0.2)},
                    {"imageId": "ortho-1", "confidence": 0.7, "bbox": wkt(0.0, 0.0, 0.5, 0.5)},
                    {"imageId": "obl-1", "confidence": 0.8, "bbox": wkt(0.3, 0.3, 0.4, 0.4)},
                    {"imageId": "ortho-1", "confidence": 0.6, "bbox": "not wkt"},
                ]}}],
            }

        def parcel(dx: float, tree: float) -> dict:
            ring = [[LON + dx - 1e-4, LAT - 1e-4], [LON + dx + 1e-4, LAT - 1e-4],
                    [LON + dx + 1e-4, LAT + 1e-4], [LON + dx - 1e-4, LAT + 1e-4],
                    [LON + dx - 1e-4, LAT - 1e-4]]
            return {"geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"parcel-tree-cover.percent": tree, "pool.detected": False,
                                   "structures": [{"footprint.area": 1000}]}}

        # Neighbour parcel first: the wrapper must pick the one containing the point.
        return {"type": "FeatureCollection", "features": [parcel(0.01, 99.0), parcel(0.0, 12.5)]}

    def request_bytes(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> bytes:
        return self.image_bytes


@pytest.fixture
def fake_vexcel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    src = tmp_path / "src.jpg"
    Image.new("RGB", (200, 200), (40, 40, 40)).save(src, "JPEG")
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    client = FakeClient(src)
    monkeypatch.setattr(vexcel, "_client", client)
    return client


def test_get_property_info_picks_containing_parcel(fake_vexcel: FakeClient) -> None:
    info = vexcel.get_property_info(LAT, LON)
    assert info.attributes["parcel-tree-cover.percent"] == 12.5
    assert "structures" not in info.attributes
    assert info.structures == [{"footprint.area": 1000}]
    assert info.raw_response["type"] == "FeatureCollection"


def test_parcel_for_point_falls_back_to_nearest() -> None:
    far = {"geometry": {"type": "Point", "coordinates": [LON + 1, LAT]}}
    near = {"geometry": {"type": "Point", "coordinates": [LON + 0.001, LAT]}}
    assert vexcel._parcel_for_point([far, near], LAT, LON) is near
    assert vexcel._parcel_for_point([], LAT, LON) is None


def test_get_aerial_returns_model(fake_vexcel: FakeClient) -> None:
    aerial = vexcel.get_aerial(LAT, LON, 7)
    assert aerial.id == "aerial-7" and Path(aerial.image_path).exists()
    assert len(aerial.metadata["bounds_lonlat"]) == 4


def test_detect_counts_and_annotates(fake_vexcel: FakeClient) -> None:
    run = vexcel_detect.detect(LAT, LON, ["parking space"], 7)
    assert run.counts == {"parking space": 2}
    first = run.detections[0]
    assert first.confidence == 0.9 and first.property_id == 7
    left, top, right, bottom = first.bbox
    assert 0 <= left < right <= 200 and 0 <= top < bottom <= 200
    assert top > 100  # south part of the square is drawn lower in the image
    assert (left, top, right, bottom) == (20.0, 160.0, 40.0, 180.0)
    assert Path(run.annotated_image_path).exists()
    json.loads(run.model_dump_json())


def test_detect_needs_prompts(fake_vexcel: FakeClient) -> None:
    with pytest.raises(ValueError):
        vexcel_detect.detect(LAT, LON, ["  "], 1)


def test_parse_detections_handles_empty() -> None:
    bounds = (0.0, 0.0, 1.0, 1.0)
    assert vexcel_detect.parse_detections({}, 1, bounds, (10, 10)) == []
    assert vexcel_detect.parse_detections({"features": [{"properties": {}}]}, 1, bounds, (10, 10)) == []


# ---------------------------------------------------------------------------
# scripts/build_sample.py
# ---------------------------------------------------------------------------


def test_build_sample_writes_json(fake_vexcel: FakeClient, capsys: pytest.CaptureFixture[str]) -> None:
    out = build_sample.main(["--n", "45", "--vexcel", "3"])
    sample = json.loads(out.read_text())
    assert sample["n"] == 45
    with_vexcel = [b for b in sample["buildings"] if b["vexcel"]]
    assert len(with_vexcel) == 3
    attrs = with_vexcel[0]["vexcel"]["attributes"]
    assert attrs["parcel-tree-cover.percent"] == 12.5
    assert attrs["pool.detected"] is False
    assert attrs["hardscapes.percent"] is None  # missing stays unknown
    assert "with Vexcel: 3" in capsys.readouterr().out


def test_build_sample_survives_vexcel_error(fake_vexcel: FakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(lat: float, lon: float) -> None:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(vexcel, "get_property_info", boom)
    sample = build_sample.build_sample(n=5, vexcel_count=2)
    assert [b["vexcel"] for b in sample["buildings"][:3]] == [
        {"error": "RuntimeError"}, {"error": "RuntimeError"}, None,
    ]


def test_detect_tiled_merges_tiles_and_dedupes(fake_vexcel: FakeClient) -> None:
    """Each tile is one detect call; the same box returned by every tile is kept once."""
    run = vexcel_detect.detect_tiled(LAT, LON, ["parking space"], 9, half_m=80.0, tile_m=80.0)
    # 2 x 2 tiles, each returning the same 2 ortho boxes -> deduped to 2.
    assert run.counts == {"parking space": 2}
    for det in run.detections:
        left, top, right, bottom = det.bbox
        assert 0 <= left < right <= 200 and 0 <= top < bottom <= 200
    assert Path(run.annotated_image_path).name == "property_9_160m_tiled_boxes.jpg"


def test_dedupe_keeps_most_confident() -> None:
    a = vexcel_detect.Detection(id="a", label="car", confidence=0.6, geometry_wkt="POLYGON((0 0,1 0,1 1,0 1,0 0))")
    b = vexcel_detect.Detection(id="b", label="car", confidence=0.9, geometry_wkt="POLYGON((0.1 0,1.1 0,1.1 1,0.1 1,0.1 0))")
    c = vexcel_detect.Detection(id="c", label="car", confidence=0.5, geometry_wkt="POLYGON((5 5,6 5,6 6,5 6,5 5))")
    assert [d.id for d in vexcel_detect._dedupe([a, b, c])] == ["b", "c"]


def test_mask_fraction_and_box_area() -> None:
    import base64
    import io

    from shapely.geometry import box as sbox

    img = Image.new("L", (10, 10), 0)
    img.paste(255, (0, 0, 10, 4))  # top 40% white
    buf = io.BytesIO()
    img.save(buf, "PNG")
    assert vexcel_detect._mask_fraction(base64.b64encode(buf.getvalue()).decode()) == 0.4
    assert vexcel_detect._mask_fraction("not base64 png") is None
    # 0.001 deg x 0.001 deg at the equator is about 111.3 m x 111.3 m.
    assert abs(vexcel_detect._box_area_m2(sbox(0, 0, 0.001, 0.001)) - 12392) < 5


def test_spaces_from_dates_needs_area() -> None:
    with pytest.raises(ValueError):
        vexcel_detect.spaces_from_dates(LAT, LON, 1, half_m=0.0, outline_wkt=None)


def test_same_spot_uses_space_dimensions() -> None:
    """Halves of one car end to end merge; side-by-side neighbours stay apart."""
    m = 1 / 111_320.0  # ~1 m in degrees near the equator

    def det(x0: float, y0: float, x1: float, y1: float) -> vexcel_detect.Detection:
        return vexcel_detect.Detection(
            id="d", label="car",
            geometry_wkt=f"POLYGON(({x0*m} {y0*m},{x1*m} {y0*m},{x1*m} {y1*m},{x0*m} {y1*m},{x0*m} {y0*m}))")

    def spot(d: vexcel_detect.Detection) -> dict:
        return {"xy": vexcel_detect._centre_m(d.geometry_wkt, 0.0), "det": d}

    car = det(0, 0, 2, 2.5)            # front half of a car pointing north
    back = det(0, 2.6, 2, 5)           # back half, touching, not overlapping
    neighbour = det(2.8, 0, 4.8, 5)    # next space, 2.8 m east
    facing = det(0, 6, 2, 11)          # head-to-head space 6 m north
    s = spot(car)
    assert vexcel_detect._same_spot(s, spot(back)["xy"], back)
    assert not vexcel_detect._same_spot(s, spot(neighbour)["xy"], neighbour)
    assert not vexcel_detect._same_spot(spot(back), spot(facing)["xy"], facing)
