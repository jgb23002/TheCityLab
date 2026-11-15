"""MatterPak point-cloud geometry: parsing math, caching and the download guards (no network)."""

import math
import zipfile

import numpy as np
import pytest

from backend import config
from backend.tools import matterport

SQ_FT_PER_SQ_M = 10.7639


def _box_points() -> list[tuple[float, float, float]]:
    """A 10 m x 4 m room rotated 30 degrees, heights spread over 0-3 m."""
    rng = np.random.default_rng(1)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
    interior = list(zip(rng.uniform(0, 10, 200), rng.uniform(0, 4, 200)))
    points = [(u * c - v * s, u * s + v * c, float(rng.uniform(0, 3))) for u, v in corners + interior]
    points += [(0.0, 0.0, 0.0), (0.0, 0.0, 3.0)]  # the full floor-to-ceiling span is present
    return points


def _zip_with_cloud(path, points, plan: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # the real downloader creates this
    lines = "\n".join(f"{x:.4f} {y:.4f} {z:.4f} 120 130 140" for x, y, z in points) + "\n"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("cloud.xyz", lines)
        bundle.writestr("model.obj", "o test\n")
        if plan:
            bundle.writestr("colorplan_000.jpg", b"\xff\xd8\xff\xd9")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    return tmp_path


def test_geometry_from_xyz_measures_height_and_rotated_footprint() -> None:
    points = np.array(_box_points(), dtype=np.float32)
    g = matterport.geometry_from_xyz(points)
    assert g["point_count"] == len(points)
    expected_sq_ft = 40 * SQ_FT_PER_SQ_M  # the rotated rectangle, not the axis-aligned box (~65 m2)
    assert abs(g["interior_footprint_sq_ft"] - expected_sq_ft) < expected_sq_ft * 0.03
    assert 2.6 * 3.28084 <= g["interior_height_ft"] <= 3.0 * 3.28084 + 0.01


def test_get_matterpak_parses_a_downloaded_bundle_and_caches(offline, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"url": 0, "download": 0}

    def fake_url(model_id: str) -> str:
        calls["url"] += 1
        return "https://example.invalid/bundle.zip"

    def fake_download(url: str, path, max_bytes: int) -> int:
        calls["download"] += 1
        _zip_with_cloud(path, _box_points())
        return path.stat().st_size

    monkeypatch.setattr(matterport, "_bundle_url", fake_url)
    monkeypatch.setattr(matterport, "_download_bundle", fake_download)
    first = matterport.get_matterpak("abc", download=True)
    assert first.available and first.point_count > 200 and first.interior_height_ft > 8
    assert first.floorplan_image_path and first.floorplan_image_path.endswith("abc.jpg")
    assert (offline / "matterport" / "bundles" / "abc.zip").exists()
    second = matterport.get_matterpak("abc")  # cached geometry, no download flag needed
    assert second == first and calls == {"url": 1, "download": 1}


def test_get_matterpak_without_download_flag_is_unavailable_and_makes_no_call(
        offline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(matterport, "_bundle_url", lambda model_id: (_ for _ in ()).throw(AssertionError("network")))
    result = matterport.get_matterpak("nope")
    assert result.available is False and "not downloaded" in result.reason


def test_size_guard_skips_oversized_bundles_and_caches_the_skip(offline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(matterport, "_bundle_url", lambda model_id: "https://example.invalid/big.zip")
    monkeypatch.setattr(matterport, "_download_bundle",
                        lambda url, path, max_bytes: (_ for _ in ()).throw(ValueError("bundle exceeds 150 MB")))
    result = matterport.get_matterpak("big", download=True)
    assert result.available is False and "exceeds 150 MB" in result.reason
    assert matterport.get_matterpak("big") == result


def test_cache_only_never_downloads(offline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    monkeypatch.setattr(matterport, "_bundle_url", lambda model_id: (_ for _ in ()).throw(AssertionError("network")))
    assert matterport.get_matterpak("x", download=True).available is False
