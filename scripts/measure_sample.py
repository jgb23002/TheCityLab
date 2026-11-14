"""Measure every building in a city sample once, resumably, so later runs never repeat paid calls.

Usage:
    python scripts/measure_sample.py --city houston|los_angeles [--property-type 11]
        [--n 50] [--workers 3] [--retry-errors] [--dry-run]

Per building (from backend.samples.sample_buildings; buildings without coordinates
are skipped and counted):
  1. Vexcel precomputed attributes  - vexcel.get_property_info (cached by backend/cache.py)
  2. aerial image                   - vexcel.get_aerial (cached)
  3. element detections             - vexcel_detect.detect for DETECT_PROMPTS (cached)
  4. vision measurement             - vision.measure_building on the aerial image

The final Measurement is saved to data/cache/vision/m-<property_id>.json in the same
format study.py's _cached_measurement reads and scripts/accuracy_check.py writes. A
building with a valid cache file is skipped. Errors and mocks are never cached. A
failed building is retried once, then recorded and the run continues. Totals and
failed ids are written to data/cache/measure_status.json (one section per city). By
default, ids that failed in an earlier run are skipped; --retry-errors tries them
again. With USE_CACHE_ONLY=true no calls are made: the run only reports what is cached.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):  # allow `python scripts/measure_sample.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config, samples, vexcel_detect  # noqa: E402
from backend.agent.registry import load_cached_measurement, redact, save_cached_measurement  # noqa: E402
from backend.schemas import Building, Measurement  # noqa: E402
from backend.tools import vexcel, vision  # noqa: E402

DETECT_PROMPTS = ["parking space", "car", "tree", "pool", "solar panel"]
MAX_WORKERS = 8
ATTEMPTS = 2  # first try plus one retry
# Upper bound on paid calls for one building with nothing cached yet: property/extract,
# ortho/extract (aerial), elements/detect, ortho/extract (detect image), one vision call.
VEXCEL_CALLS_PER_BUILDING = 4
MODEL_CALLS_PER_BUILDING = 1


def status_path() -> Path:
    """Return data/cache/measure_status.json (resolved at call time for tests)."""
    return config.CACHE_DIR / "measure_status.json"


def _short_reason(exc: BaseException | str) -> str:
    """A one-line, credential-free failure reason."""
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    text = " ".join(str(redact(text)).split())
    return text[:120]


def _previous_failures(city: str) -> set[int]:
    """Ids recorded as failed for *city* in the last status file, if any."""
    try:
        data = json.loads(status_path().read_text(encoding="utf-8"))
        return {int(i) for i in data.get(city, {}).get("failed_ids", [])}
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def measure_one(building: Building) -> Measurement:
    """Run the four steps for one building and return a successful real Measurement.

    Raises on any failure, including a vision result with observations.error, so the
    caller can retry. Every Vexcel request goes through the cached, 429-retrying client.
    """
    lat, lon, pid = building.latitude, building.longitude, building.id
    info = vexcel.get_property_info(lat, lon)
    image = vexcel.get_aerial(lat, lon, pid)
    if not image.image_path:
        raise RuntimeError("get_aerial returned no image")
    detections = vexcel_detect.detect(lat, lon, DETECT_PROMPTS, pid)
    measurement = vision.measure_building(image.image_path, pid)
    observations = dict(measurement.observations or {})
    if observations.get("error"):
        raise RuntimeError(f"vision: {observations['error']}")
    if observations.get("is_mock"):
        raise RuntimeError("vision returned a mock measurement")
    if measurement.property_id != pid:
        raise RuntimeError(f"vision returned property_id {measurement.property_id}")
    # Vexcel's boxes are in the detect image's pixel frame (no buffer), not the aerial
    # image's, so they are summarised in observations rather than mixed into detections.
    observations.update({
        "vexcel_property_info_id": info.id,
        "vexcel_detect_counts": dict(detections.counts),
        "vexcel_detect_annotated_image": detections.annotated_image_path,
        "measured_by": "scripts/measure_sample.py",
    })
    return measurement.model_copy(update={"observations": observations})


def _process(building: Building, measure: Callable[[Building], Measurement]) -> tuple[str, str]:
    """Measure with one retry and cache the result: returns (status, reason)."""
    reason = ""
    for _ in range(ATTEMPTS):
        try:
            measurement = measure(building)
        except Exception as exc:  # any provider, image or validation error
            reason = _short_reason(exc)
            continue
        if not save_cached_measurement(measurement):
            reason = "measurement not cacheable (error or mock)"
            continue
        return "ok", ""
    return "failed", reason


def run(
    city: str, property_type: int = 11, n: int = 50, workers: int = 3, retry_errors: bool = False,
    dry_run: bool = False, measure: Callable[[Building], Measurement] = measure_one,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure a city sample and return the summary (also written to measure_status.json)."""
    buildings = samples.sample_buildings(city, property_type, n=n)
    city_key = samples.CITY_FILES.get(city.strip().lower(), city)
    no_coords = [b for b in buildings if b.latitude is None or b.longitude is None]
    located = [b for b in buildings if b not in no_coords]
    previous = set() if retry_errors else _previous_failures(city_key)
    total = len(buildings)

    summary: dict[str, Any] = {"city": city_key, "property_type": property_type, "sampled": total,
                               "ok": 0, "cached": 0, "failed": 0, "skipped": len(no_coords),
                               "skipped_no_coordinates": [b.id for b in no_coords],
                               "skipped_previous_failure": [], "failed_ids": [], "failures": {}}
    todo: list[Building] = []
    for b in located:
        if load_cached_measurement(b.id) is not None:
            summary["cached"] += 1
        elif b.id in previous:
            summary["skipped"] += 1
            summary["skipped_previous_failure"].append(b.id)
        else:
            todo.append(b)

    if dry_run:
        summary.update({
            "dry_run": True, "would_measure": [b.id for b in todo],
            "estimated_vexcel_calls_max": len(todo) * VEXCEL_CALLS_PER_BUILDING,
            "estimated_model_calls": len(todo) * MODEL_CALLS_PER_BUILDING,
        })
        for b in todo:
            out(f"would measure property {b.id} ({b.latitude:.5f}, {b.longitude:.5f})")
        out(f"DRY RUN: {len(todo)} to measure, {summary['cached']} cached, {summary['skipped']} skipped; "
            f"up to {summary['estimated_vexcel_calls_max']} Vexcel calls (fewer if already cached) "
            f"and {summary['estimated_model_calls']} vision model calls. No calls made.")
        return summary

    lock = threading.Lock()
    done = [0]

    def log(building: Building, status: str, reason: str = "") -> None:
        """Print one progress line per building, in completion order."""
        with lock:
            done[0] += 1
            out(f"{done[0]}/{total} property {building.id} {status}" + (f": {reason}" if reason else ""))

    for b in no_coords:
        log(b, "skipped", "no latitude/longitude")
    for b in located:
        if b in todo:
            continue
        if b.id in summary["skipped_previous_failure"]:
            log(b, "skipped", "failed in an earlier run (use --retry-errors)")
        else:
            log(b, "cached")

    if config.USE_CACHE_ONLY:
        summary["missing_from_cache"] = [b.id for b in todo]
        for b in todo:
            log(b, "missing", "not cached and USE_CACHE_ONLY=true")
    else:
        def work(building: Building) -> None:
            status, reason = _process(building, measure)
            with lock:
                summary[status] += 1
                if status == "failed":
                    summary["failed_ids"].append(building.id)
                    summary["failures"][str(building.id)] = reason
            log(building, status, reason)

        with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
            list(pool.map(work, todo))

    summary["failed_ids"].sort()
    _write_status(city_key, summary)
    out(f"TOTALS: ok {summary['ok']}, cached {summary['cached']}, failed {summary['failed']}, "
        f"skipped {summary['skipped']}" + (f", missing {len(summary['missing_from_cache'])} (cache-only)"
                                         if "missing_from_cache" in summary else ""))
    out(f"Status written to {status_path()}")
    return summary


def _write_status(city_key: str, summary: dict[str, Any]) -> None:
    """Update this city's section of measure_status.json, keeping other cities."""
    path = status_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[city_key] = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Parse the CLI and measure one city sample."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", required=True, choices=["houston", "los_angeles"])
    ap.add_argument("--property-type", type=int, default=11)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3, help=f"parallel buildings (1-{MAX_WORKERS})")
    ap.add_argument("--retry-errors", action="store_true", help="retry ids that failed in an earlier run")
    ap.add_argument("--dry-run", action="store_true", help="list what would be measured; make no calls")
    args = ap.parse_args(argv)
    summary = run(args.city, args.property_type, args.n, args.workers, args.retry_errors, args.dry_run)
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
