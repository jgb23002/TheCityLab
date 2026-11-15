"""Measure every sampled building from Vexcel for Q1/Q2 (and roof area for Q3).

Run ``python scripts/measure_vexcel.py`` (``--city``, ``--n``, ``--refresh``).

For each building in ``data/samples/<city>.csv``:

1. Find the Vexcel parcel matching the CoStar property (area closest to CoStar
   ``lotSize``); CoStar's point often sits in a small neighbouring parcel.
   When no parcel matches, the values are left unknown (the parcel at the
   point is often a neighbour, a park or a road); its raw numbers are kept
   in ``observations`` for reference only. Points Vexcel does not cover are
   recorded as "no Vexcel coverage".
2. Read Vexcel's precomputed measurements for that parcel:
   ``parcel-tree-cover.percent`` -> green cover (Q1),
   ``hardscapes.percent`` -> paved surface (Q2),
   sum of structure ``footprint.area`` -> roof area in sq ft (Q3 solar).
3. Save an aerial photo centred on the parcel for the website cards.
4. Write a ``Measurement`` to ``data/cache/vision/m-<id>.json``, the file
   ``backend/agent/study.py`` reads, so the agent answers offline from cache.
   An existing Sky Inspector (Claude) measurement there is kept: Vexcel's
   numbers fill green/paved/roof, Claude's parking count and boxes stay, and
   Claude's own estimates are kept in ``observations`` as a second opinion.

With ``--detect``, Vexcel's AI also boxes the cars on one ~80 m photo
centred on the parcel; the boxes go into ``Measurement.detections`` and a
boxed copy of the photo is saved (about 2 more calls per building). Car
counts are not parking-space counts (see docs/PARKING_METHOD.md).

Resumable: a building already in ``data/cache/measurements/<city>/`` is
skipped unless ``--refresh``. About 2 Vexcel calls per new building, all
cached. The console shows only counts, never keys or response bodies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapely import wkt as shapely_wkt  # noqa: E402

from backend import config, vexcel_detect  # noqa: E402
from backend.samples import sample_buildings  # noqa: E402
from backend.schemas import Building, Detection, Measurement  # noqa: E402
from backend.tools import vexcel  # noqa: E402


SOURCE = "Vexcel precomputed parcel attributes"


def _num(value: Any) -> float | None:
    """A finite number or None."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def lot_sizes(ids: list[int]) -> dict[int, float]:
    """CoStar lotSize in acres for *ids* (empty when the kit is not unzipped)."""
    path = config.COSTAR_DATA_DIR / "commercial-property.jsonl"
    if not path.exists():
        return {}
    from backend.tools.costar_data import _load_jsonl

    df = _load_jsonl("commercial-property.jsonl")
    df = df[df["id"].isin(ids)]
    out: dict[int, float] = {}
    for pid, size, unit in zip(df["id"], df.get("lotSize"), df.get("lotSizeMeasurementType")):
        size = _num(size)
        if size is None or size <= 0:
            continue
        if unit == 2:  # acres
            out[int(pid)] = size
        elif unit == 1:  # square feet
            out[int(pid)] = size / 43_560.0
    return out


def parcel_values(props: dict[str, Any]) -> dict[str, Any]:
    """Green %, paved %, roof sq ft and a few context fields from Vexcel attributes."""
    footprints = [
        _num(s.get("footprint.area")) for s in props.get("structures") or [] if isinstance(s, dict)
    ]
    footprints = [f for f in footprints if f is not None and f > 0]
    green = _num(props.get("parcel-tree-cover.percent"))
    paved = _num(props.get("hardscapes.percent"))
    return {
        "green_cover_percent": green if green is not None and 0 <= green <= 100 else None,
        "paved_surface_percent": paved if paved is not None and 0 <= paved <= 100 else None,
        "roof_area_sq_ft": round(sum(footprints), 1) if footprints else None,
        "structure_count": len(footprints),
        "pool_detected": props.get("pool.detected"),
        "capture_date": props.get("metadata.ortho-date-estimate"),
    }


def measure_one(building: Building, lot_acres: float | None) -> dict[str, Any]:
    """Measure one building; return a JSON-ready record (never raises)."""
    lat, lon = building.latitude, building.longitude
    record: dict[str, Any] = {"id": building.id, "lot_acres": lot_acres}
    try:
        parcel = vexcel_detect.property_parcel(lat, lon, lot_acres) if lot_acres else None
        if parcel is not None:
            props, basis = parcel["properties"], "parcel matching CoStar lotSize"
            centre = shapely_wkt.loads(parcel["wkt"]).centroid
            record.update(parcel_acres=round(parcel["acres"], 2),
                          parcel_area_error=round(parcel["area_error"], 3))
            img_lat, img_lon = centre.y, centre.x
        else:
            info = vexcel.get_property_info(lat, lon)
            props = {**(info.attributes or {}), "structures": info.structures or []}
            basis = "parcel at CoStar point (no parcel matched lotSize)"
            img_lat, img_lon = lat, lon
        record.update(parcel_values(props), basis=basis, image_center=[img_lat, img_lon])
        record["image_path"] = vexcel.get_aerial(img_lat, img_lon, building.id).image_path
    except Exception as exc:  # one bad building must not stop the run
        if getattr(getattr(exc, "response", None), "status_code", None) == 404:
            record.update(basis="no Vexcel coverage", green_cover_percent=None,
                          paved_surface_percent=None, roof_area_sq_ft=None)
        else:
            record["error"] = type(exc).__name__
    return record


def detect_cars(property_id: int, lat: float, lon: float) -> dict[str, Any]:
    """Box cars on one ~80 m photo; return a JSON-ready summary (never raises)."""
    try:
        run = vexcel_detect.detect(lat, lon, ["car"], property_id)
    except Exception as exc:
        return {"error": type(exc).__name__}
    return {
        "car_count": run.counts.get("car", 0),
        "image_path": run.image_path,
        "annotated_image_path": run.annotated_image_path,
        "detections": [d.model_dump(mode="json") for d in run.detections],
    }


def to_measurement(record: dict[str, Any], existing: Measurement | None) -> Measurement:
    """Build the Measurement the study pipeline reads, merging any Claude result."""
    obs: dict[str, Any] = {
        "source": SOURCE,
        "area_basis": record.get("basis"),
        "lot_acres": record.get("lot_acres"),
        "parcel_acres": record.get("parcel_acres"),
        "structure_count": record.get("structure_count"),
        "capture_date": record.get("capture_date"),
        "image_path": record.get("image_path"),
        "roof_area_note": "sum of Vexcel structure footprints; not pitched roof surface",
    }
    keys = ("green_cover_percent", "paved_surface_percent", "roof_area_sq_ft")
    fields = {k: record.get(k) for k in keys}
    if not str(record.get("basis", "")).startswith("parcel matching"):
        # Unmatched parcel: values may describe a neighbour; keep them as notes only.
        obs["unmatched_parcel_values"] = fields
        fields = {k: None for k in keys}
    if existing is not None:
        prev = existing.observations or {}
        obs["sky_inspector"] = {
            "green_cover_percent": existing.green_cover_percent,
            "paved_surface_percent": existing.paved_surface_percent,
            "roof_area_sq_ft": existing.roof_area_sq_ft,
            "notes": prev.get("notes"),
        }
        return existing.model_copy(update={
            **{k: v if v is not None else getattr(existing, k) for k, v in fields.items()},
            "observations": {**prev, **obs},
        })
    det = record.get("detection") or {}
    if det:
        obs.update(car_count=det.get("car_count"), detection_image_path=det.get("image_path"),
                   annotated_image_path=det.get("annotated_image_path"),
                   detection_note="cars on one capture; not a parking-space count")
    detections = [Detection.model_validate(d) for d in det.get("detections") or []] or None
    return Measurement(id=f"m-{record['id']}", property_id=record["id"], observations=obs,
                       detections=detections, **fields)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Measure the sample, write cache files and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", default="houston")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--refresh", action="store_true", help="re-measure buildings already done")
    parser.add_argument("--detect", action="store_true", help="also box cars on one photo per building")
    args = parser.parse_args(argv)

    buildings = sample_buildings(args.city, 11, n=args.n)
    lots = lot_sizes([b.id for b in buildings])
    rec_dir = config.CACHE_DIR / "measurements" / args.city
    vision_dir = config.CACHE_DIR / "vision"
    rec_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    counts = {"buildings": len(buildings), "measured_now": 0, "from_cache": 0, "errors": 0,
              "matched_parcel": 0, "green": 0, "paved": 0, "roof": 0}
    for b in buildings:
        rec_path = rec_dir / f"{b.id}.json"
        if rec_path.exists() and not args.refresh:
            record = json.loads(rec_path.read_text(encoding="utf-8"))
            counts["from_cache"] += 1
        else:
            record = measure_one(b, lots.get(b.id))
            if "error" not in record:
                rec_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            counts["measured_now"] += 1
        if "error" in record:
            counts["errors"] += 1
            continue
        if args.detect and record.get("image_center") and "detection" not in record:
            record["detection"] = detect_cars(b.id, *record["image_center"])
            rec_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if (record.get("detection") or {}).get("car_count") is not None:
            counts["with_boxes"] = counts.get("with_boxes", 0) + 1
        matched = record.get("basis", "").startswith("parcel matching")
        counts["matched_parcel"] += matched
        counts["green"] += matched and record.get("green_cover_percent") is not None
        counts["paved"] += matched and record.get("paved_surface_percent") is not None
        counts["roof"] += matched and record.get("roof_area_sq_ft") is not None

        m_path = vision_dir / f"m-{b.id}.json"
        existing = None
        if m_path.exists():
            try:
                existing = Measurement.model_validate_json(m_path.read_text(encoding="utf-8"))
            except ValueError:
                existing = None
            # Merge only with a Sky Inspector (Claude) result, never our own earlier file.
            if existing is not None and (existing.observations or {}).get("source") == SOURCE:
                existing = None
        m_path.write_text(to_measurement(record, existing).model_dump_json(), encoding="utf-8")

    print(" | ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"saved: {rec_dir} and {vision_dir}")
    return counts


if __name__ == "__main__":
    main()
