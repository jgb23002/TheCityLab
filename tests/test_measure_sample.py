"""scripts/measure_sample.py with mocked tools: no network, keys or model calls."""

import json
import socket
from typing import Any

import pytest

from backend import config, samples, vexcel_detect
from backend.agent.study import _cached_measurement
from backend.schemas import AerialImage, Building, Measurement, VexcelPropertyInfo
from backend.tools import vexcel, vision
from scripts import measure_sample as ms

MATCHED_PARCEL = {
    "wkt": "POLYGON((-95.41 29.70, -95.41 29.71, -95.40 29.71, -95.40 29.70, -95.41 29.70))",
    "properties": {
        "structures": [{"footprint.area": 1200.0}, {"footprint.area": 800.5}, {"footprint.area": None}],
        "parcel-tree-cover.percent": 12.5,
        "hardscapes.percent": 40.0,
        "pool.detected": False,
    },
    "acres": 1.02,
    "area_error": 0.048,
}


def b(pid: int, located: bool = True) -> Building:
    """A sample building, optionally without coordinates."""
    return Building(id=pid, market_id=14, property_type=11,
                    latitude=29.7 if located else None, longitude=-95.4 if located else None)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, Any]:
    """Patch the sample and every tool step; count calls; fail any network use."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    state: dict[str, Any] = {"calls": {"info": 0, "aerial": 0, "detect": 0, "vision": 0, "parcel": 0},
                             "buildings": [b(1), b(2), b(3, located=False), b(4)],
                             "vision_error": set(), "raise_on": set(), "flaky": set(),
                             "lot_acres": {}, "parcel": None}

    monkeypatch.setattr(samples, "sample_buildings", lambda city, property_type=11, outcome=None, n=50:
                        state["buildings"][:n])
    # The CoStar kit is not read in tests; a building has a lot size only when the test says so.
    monkeypatch.setattr(ms, "lot_sizes", lambda ids: {pid: state["lot_acres"][pid] for pid in ids if pid in state["lot_acres"]})

    def parcel(lat: float, lon: float, lot_acres: float, search_half_m=None):
        state["calls"]["parcel"] += 1
        return state["parcel"]

    def info(lat: float, lon: float) -> VexcelPropertyInfo:
        state["calls"]["info"] += 1
        return VexcelPropertyInfo(id="vx-1", latitude=lat, longitude=lon)

    def aerial(lat: float, lon: float, property_id: int) -> AerialImage:
        state["calls"]["aerial"] += 1
        if property_id in state["raise_on"]:
            raise RuntimeError(f"HTTP 500 for property {property_id}")
        return AerialImage(id=f"a{property_id}", property_id=property_id, image_path=f"/img/{property_id}.jpg",
                           latitude=lat, longitude=lon)

    def detect(lat: float, lon: float, prompts: list[str], property_id: int) -> vexcel_detect.DetectionRun:
        state["calls"]["detect"] += 1
        return vexcel_detect.DetectionRun(property_id=property_id, prompts=prompts,
                                          counts={p: 1 for p in prompts}, detections=[],
                                          image_path="d.jpg", annotated_image_path="d_boxes.jpg")

    def measure(image_path: str, property_id: int) -> Measurement:
        state["calls"]["vision"] += 1
        if property_id in state["flaky"]:
            state["flaky"].discard(property_id)
            raise TimeoutError("vision timed out")
        obs = {"error": "vision call failed: Timeout"} if property_id in state["vision_error"] else {"notes": "ok"}
        return Measurement(id=f"m-{property_id}", property_id=property_id, parking_space_count=10,
                           roof_area_sq_ft=5000.0, green_cover_percent=30.0, observations=obs)

    monkeypatch.setattr(vexcel_detect, "property_parcel", parcel)
    monkeypatch.setattr(vexcel, "get_property_info", info)
    monkeypatch.setattr(vexcel, "get_aerial", aerial)
    monkeypatch.setattr(vexcel_detect, "detect", detect)
    monkeypatch.setattr(vision, "measure_building", measure)
    state["tmp"] = tmp_path
    return state


def run(lines: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Run the script's main loop, collecting its output lines."""
    sink = lines if lines is not None else []
    return ms.run("houston", out=sink.append, **kwargs)


def test_measures_all_and_writes_the_study_cache_format(tools: dict[str, Any]) -> None:
    """Each located building runs all four steps once; files validate like study.py reads them."""
    lines: list[str] = []
    summary = run(lines)
    assert (summary["ok"], summary["cached"], summary["failed"], summary["skipped"]) == (3, 0, 0, 1)
    assert tools["calls"] == {"info": 3, "aerial": 3, "detect": 3, "vision": 3, "parcel": 0}
    for pid in (1, 2, 4):
        cached = _cached_measurement(b(pid))
        assert cached is not None and cached.property_id == pid
        assert cached.observations["vexcel_detect_counts"]["parking space"] == 1
    assert summary["skipped_no_coordinates"] == [3]
    assert sum(1 for line in lines if line.split(" ")[0].endswith("/4")) == 4
    assert any(line.startswith("TOTALS: ok 3, cached 0, enriched 0, failed 0, skipped 1") for line in lines)
    status = json.loads((tools["tmp"] / "measure_status.json").read_text())
    assert status["houston"]["ok"] == 3 and status["houston"]["failed_ids"] == []


def test_unmatched_parcel_keeps_vision_values_untagged(tools: dict[str, Any]) -> None:
    """Without a lotSize match the vision roof stays, with no real-measurement source tag."""
    run(n=1)
    cached = _cached_measurement(b(1))
    assert cached.roof_area_sq_ft == 5000.0 and cached.parking_space_count == 10
    obs = cached.observations
    assert "source" not in obs
    assert obs["area_basis"].startswith("parcel at CoStar point")
    assert obs["unmatched_parcel_values"] == {"green_cover_percent": None, "paved_surface_percent": None,
                                              "roof_area_sq_ft": None}
    assert obs["sky_inspector"]["roof_area_sq_ft"] == 5000.0


def test_matched_parcel_fills_roof_green_paved_and_tags_the_source(tools: dict[str, Any]) -> None:
    """A lotSize-matched Vexcel parcel supplies the three measured fields; vision keeps parking."""
    tools["lot_acres"][1] = 1.0
    tools["parcel"] = MATCHED_PARCEL
    run(n=1)
    assert tools["calls"]["parcel"] == 1
    cached = _cached_measurement(b(1))
    assert cached.roof_area_sq_ft == 2000.5  # sum of the two real footprints
    assert cached.green_cover_percent == 12.5 and cached.paved_surface_percent == 40.0
    assert cached.parking_space_count == 10  # vision's count is untouched
    obs = cached.observations
    assert obs["source"] == ms.SOURCE == "Vexcel precomputed parcel attributes"
    assert obs["area_basis"] == "parcel matching CoStar lotSize"
    assert obs["parcel_acres"] == 1.02 and obs["structure_count"] == 2
    assert obs["sky_inspector"] == {"green_cover_percent": 30.0, "paved_surface_percent": None,
                                    "roof_area_sq_ft": 5000.0}
    # The aerial image and detections are centred on the matched parcel, not CoStar's point.
    assert obs["image_center"] != [29.7, -95.4]


@pytest.mark.parametrize(
    ("props", "expected"),
    [
        ({}, {"green_cover_percent": None, "paved_surface_percent": None, "roof_area_sq_ft": None, "structure_count": 0}),
        ({"structures": [{"footprint.area": 10.25}, {"footprint.area": 0}, "bad"], "parcel-tree-cover.percent": 101,
          "hardscapes.percent": 55.5},
         {"green_cover_percent": None, "paved_surface_percent": 55.5, "roof_area_sq_ft": 10.2, "structure_count": 1}),
    ],
)
def test_parcel_values_sums_footprints_and_checks_ranges(props: dict, expected: dict) -> None:
    """Footprints sum (zeros and junk ignored); out-of-range percentages become unknown."""
    values = ms.parcel_values(props)
    assert {k: values[k] for k in expected} == expected


def test_enrich_parcels_upgrades_cached_vision_only_measurements_without_vision_calls(tools: dict[str, Any]) -> None:
    """--enrich-parcels adds the Vexcel footprint, green and paved to cached measurements that lack them."""
    run(n=1)  # vision-only cache for building 1 (no parcel matched yet)
    assert "source" not in _cached_measurement(b(1)).observations
    tools["lot_acres"][1] = 1.0
    tools["parcel"] = MATCHED_PARCEL
    tools["calls"] = {k: 0 for k in tools["calls"]}
    lines: list[str] = []
    summary = run(lines, n=1, enrich=True)
    assert summary["enriched"] == 1 and summary["cached"] == 0 and summary["ok"] == 0
    assert tools["calls"]["vision"] == 0 and tools["calls"]["parcel"] == 1
    cached = _cached_measurement(b(1))
    assert cached.roof_area_sq_ft == 2000.5 and cached.parking_space_count == 10
    assert cached.observations["source"] == ms.SOURCE
    assert cached.observations["sky_inspector"]["roof_area_sq_ft"] == 5000.0  # vision's reading kept
    assert cached.observations["enriched_by"].endswith("--enrich-parcels")
    assert any(line.endswith("property 1 enriched") for line in lines)
    again = run(n=1, enrich=True)  # already tagged: nothing left to enrich
    assert again["enriched"] == 0 and again["cached"] == 1


def test_resume_skips_cached_buildings(tools: dict[str, Any]) -> None:
    """A second run makes no calls: every building is reported cached."""
    run()
    tools["calls"] = {k: 0 for k in tools["calls"]}
    lines: list[str] = []
    summary = run(lines)
    assert (summary["ok"], summary["cached"], summary["skipped"]) == (0, 3, 1)
    assert tools["calls"] == {"info": 0, "aerial": 0, "detect": 0, "vision": 0, "parcel": 0}
    assert any(line.endswith("property 1 cached") for line in lines)


def test_errors_are_not_cached_and_failure_does_not_stop_the_run(tools: dict[str, Any]) -> None:
    """A vision error is retried once, never cached, recorded; other buildings still finish."""
    tools["vision_error"].add(2)
    tools["raise_on"].add(4)
    lines: list[str] = []
    summary = run(lines, workers=2)
    assert (summary["ok"], summary["failed"]) == (1, 2)
    assert summary["failed_ids"] == [2, 4]
    assert "vision call failed" in summary["failures"]["2"] and "HTTP 500" in summary["failures"]["4"]
    assert not (tools["tmp"] / "vision" / "m-2.json").exists()
    assert not (tools["tmp"] / "vision" / "m-4.json").exists()
    assert (tools["tmp"] / "vision" / "m-1.json").exists()
    assert tools["calls"]["vision"] == 1 + 2  # building 1 once, building 2 twice (one retry)
    assert any("property 2 failed: RuntimeError: vision: vision call failed" in line for line in lines)
    status = json.loads((tools["tmp"] / "measure_status.json").read_text())
    assert status["houston"]["failed_ids"] == [2, 4]


def test_one_retry_recovers_a_transient_failure(tools: dict[str, Any]) -> None:
    """A building that fails once and then succeeds counts as ok."""
    tools["flaky"].add(1)
    summary = run()
    assert summary["ok"] == 3 and summary["failed"] == 0
    assert (tools["tmp"] / "vision" / "m-1.json").exists()


def test_previous_failures_are_skipped_unless_retry_errors(tools: dict[str, Any]) -> None:
    """Failed ids from the status file are skipped by default and retried with --retry-errors."""
    tools["vision_error"].add(2)
    run()
    tools["vision_error"].clear()
    again = run()
    assert again["skipped_previous_failure"] == [2] and again["skipped"] == 2 and again["ok"] == 0
    retried = run(retry_errors=True)
    assert retried["ok"] == 1 and retried["failed"] == 0 and retried["cached"] == 2


def test_mock_measurements_are_never_cached(tools: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A measurement marked is_mock is treated as a failure, not cached."""
    monkeypatch.setattr(vision, "measure_building", lambda image_path, property_id: Measurement(
        id="m", property_id=property_id, observations={"is_mock": True}))
    summary = run()
    assert summary["ok"] == 0 and summary["failed"] == 3
    assert not (tools["tmp"] / "vision").exists() or not any((tools["tmp"] / "vision").iterdir())


def test_cache_only_makes_no_calls(tools: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """USE_CACHE_ONLY reports cached vs missing and calls nothing."""
    run(n=1)  # cache building 1
    tools["calls"] = {k: 0 for k in tools["calls"]}
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    lines: list[str] = []
    summary = run(lines)
    assert tools["calls"] == {"info": 0, "aerial": 0, "detect": 0, "vision": 0, "parcel": 0}
    assert summary["cached"] == 1 and summary["missing_from_cache"] == [2, 4]
    assert (summary["ok"], summary["failed"], summary["skipped"]) == (0, 0, 1)
    assert any("missing 2 (cache-only)" in line for line in lines)
    assert any(line.endswith("property 2 missing: not cached and USE_CACHE_ONLY=true") for line in lines)


def test_dry_run_estimates_and_makes_no_calls(tools: dict[str, Any]) -> None:
    """--dry-run lists the buildings to measure and estimates paid calls, without calling."""
    run(n=1)
    tools["calls"] = {k: 0 for k in tools["calls"]}
    lines: list[str] = []
    summary = run(lines, dry_run=True)
    assert tools["calls"] == {"info": 0, "aerial": 0, "detect": 0, "vision": 0, "parcel": 0}
    assert summary["would_measure"] == [2, 4] and summary["cached"] == 1
    assert summary["estimated_vexcel_calls_max"] == 2 * ms.VEXCEL_CALLS_PER_BUILDING
    assert summary["estimated_model_calls"] == 2
    assert lines[-1].startswith("DRY RUN: 2 to measure, 0 to enrich, 1 cached, 1 skipped")


def test_failure_reasons_never_contain_secrets(tools: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Secret values in exception text are redacted in logs and the status file."""
    monkeypatch.setenv("VEXCEL_API_KEY", "private-test-value")

    def leaky(lat: float, lon: float) -> VexcelPropertyInfo:
        raise RuntimeError("bad request with key private-test-value")

    monkeypatch.setattr(vexcel, "get_property_info", leaky)
    lines: list[str] = []
    run(lines)
    text = "\n".join(lines) + (tools["tmp"] / "measure_status.json").read_text()
    assert "private-test-value" not in text and "[REDACTED]" in text


def test_cli_parses_arguments(tools: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    """main() runs the requested city and returns non-zero only when something failed."""
    assert ms.main(["--city", "houston", "--n", "2", "--workers", "1"]) == 0
    assert "TOTALS: ok 2" in capsys.readouterr().out
