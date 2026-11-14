"""Tests for the team-plan data functions: sample_buildings, run_regression,
vexcel.detect / get_precomputed, and scripts/build_sample.py. No network."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import backend.config as cfg
from backend.tools import costar_data, stats, vexcel
from scripts import build_sample

# ---------------------------------------------------------------------------
# costar_data.sample_buildings (uses the committed data/samples CSVs)
# ---------------------------------------------------------------------------


def test_sample_buildings_houston_has_q1_fields() -> None:
    rows = costar_data.sample_buildings("houston", 11, outcome="max_rent", n=50)
    assert 40 <= len(rows) <= 50
    for r in rows:
        assert isinstance(r["id"], int)
        assert r["latitude"] is not None and r["longitude"] is not None
        assert r["max_rent"] is not None
        assert {"total_num_units", "year_built"} <= r.keys()
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    json.dumps(rows)  # JSON-serialisable


def test_sample_buildings_aliases_and_limit() -> None:
    assert len(costar_data.sample_buildings("Los Angeles", n=5)) == 5
    assert costar_data.sample_buildings("LA", n=0) == []


def test_sample_buildings_blank_outcome_dropped() -> None:
    all_la = costar_data.sample_buildings("los_angeles", n=100)
    with_parking = costar_data.sample_buildings("los_angeles", outcome="surface_parking_spaces", n=100)
    assert all(r["surface_parking_spaces"] is not None for r in with_parking)
    assert len(with_parking) <= len(all_la)


def test_sample_buildings_bad_input() -> None:
    with pytest.raises(ValueError):
        costar_data.sample_buildings("paris")
    with pytest.raises(ValueError):
        costar_data.sample_buildings("houston", outcome="not_a_column")


# ---------------------------------------------------------------------------
# stats.run_regression
# ---------------------------------------------------------------------------


def test_run_regression_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "id": range(n),
        "green": rng.uniform(0, 40, n),
        "total_num_units": rng.integers(50, 500, n),
        "year_built": rng.integers(1960, 2020, n),
    })
    df["max_rent"] = 1000 + 10 * df["green"] + rng.normal(0, 20, n)
    result = stats.run_regression(df, "max_rent", "green")
    assert result["n"] == n
    assert result["controls"] == ["total_num_units", "year_built"]
    low, high = result["conf_int_95"]
    assert low < 10 < high
    assert result["p_value"] < 0.001
    assert "associated" in result["finding"]


# ---------------------------------------------------------------------------
# vexcel.detect / get_precomputed
# ---------------------------------------------------------------------------

LAT, LON = 29.76, -95.39


class FakeClient:
    """Stands in for the HTTP client; records requests, returns canned data."""

    def __init__(self, image_path: Path) -> None:
        self.calls: list[tuple[str, str]] = []
        self.image_bytes = image_path.read_bytes()

    def request_json(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 30) -> dict:
        self.calls.append((method, endpoint))
        if endpoint == vexcel.DETECT_ENDPOINT:
            assert payload["prompts"] == ["parking space"]
            assert payload["wkt"].startswith("POLYGON ((-95.")  # longitude first
            min_lon, min_lat, max_lon, max_lat = vexcel._square_bounds(LAT, LON)
            w, h = max_lon - min_lon, max_lat - min_lat
            square = [[min_lon + w * 0.1, min_lat + h * 0.1], [min_lon + w * 0.2, min_lat + h * 0.1],
                      [min_lon + w * 0.2, min_lat + h * 0.2], [min_lon + w * 0.1, min_lat + h * 0.2]]
            return {"features": [
                {"geometry": {"type": "Polygon", "coordinates": [square]}, "properties": {"score": 0.9}},
                {"bbox": [min_lon, min_lat, min_lon + w / 2, min_lat + h / 2], "properties": {}},
            ]}
        def parcel(dx: float, tree: float) -> dict:
            ring = [[LON + dx - 1e-4, LAT - 1e-4], [LON + dx + 1e-4, LAT - 1e-4],
                    [LON + dx + 1e-4, LAT + 1e-4], [LON + dx - 1e-4, LAT + 1e-4],
                    [LON + dx - 1e-4, LAT - 1e-4]]
            return {"geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"parcel-tree-cover.percent": tree, "pool.detected": False}}

        # Neighbour parcel first: the wrapper must pick the one containing the point.
        return {"features": [parcel(0.01, 99.0), parcel(0.0, 12.5)]}

    def request_bytes(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> bytes:
        self.calls.append((method, endpoint))
        return self.image_bytes


@pytest.fixture
def fake_vexcel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    src = tmp_path / "src.jpg"
    Image.new("RGB", (200, 200), (40, 40, 40)).save(src, "JPEG")
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    client = FakeClient(src)
    monkeypatch.setattr(vexcel, "_client", client)
    return client


def test_detect_counts_and_annotates(fake_vexcel: FakeClient) -> None:
    result = vexcel.detect(LAT, LON, ["parking space"], property_id=7)
    assert result["counts"] == {"parking space": 2}
    assert len(result["boxes"]) == 2
    first = result["boxes"][0]
    assert first["score"] == 0.9
    x0, y0, x1, y1 = first["bbox_px"]
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 200
    assert y0 > 100  # south part of the square is drawn lower in the image
    annotated = Path(result["annotated_image_path"])
    assert annotated.exists() and annotated.name == "property_7.jpg"
    json.dumps(result)


def test_detect_needs_prompts(fake_vexcel: FakeClient) -> None:
    with pytest.raises(ValueError):
        vexcel.detect(LAT, LON, ["  "])


def test_parse_detections_handles_empty() -> None:
    assert vexcel.parse_detections({}, ["pool"]) == []
    assert vexcel.parse_detections({"features": [{"properties": {}}]}, ["pool"]) == []


def test_parcel_for_point_falls_back_to_nearest() -> None:
    far = {"geometry": {"type": "Point", "coordinates": [LON + 1, LAT]}, "properties": {"n": 1}}
    near = {"geometry": {"type": "Point", "coordinates": [LON + 0.001, LAT]}, "properties": {"n": 2}}
    assert vexcel._parcel_for_point([far, near], LAT, LON) is near
    assert vexcel._parcel_for_point([], LAT, LON) is None


def test_get_precomputed_flattens(fake_vexcel: FakeClient) -> None:
    info = vexcel.get_precomputed(LAT, LON)
    assert info["attributes"]["parcel-tree-cover.percent"] == 12.5


# ---------------------------------------------------------------------------
# scripts/build_sample.py
# ---------------------------------------------------------------------------


def test_build_sample_writes_json(fake_vexcel: FakeClient, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
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
    def boom(lat: float, lon: float) -> dict:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(vexcel, "get_precomputed", boom)
    sample = build_sample.build_sample(n=5, vexcel_count=2)
    assert [b["vexcel"] for b in sample["buildings"][:3]] == [
        {"error": "RuntimeError"}, {"error": "RuntimeError"}, None,
    ]
