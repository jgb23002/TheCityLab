"""Vexcel custom-element detection: find things like "parking space" in an image.

A data helper for ``backend.tools.vision`` (not an agent tool itself). Returns
:class:`backend.schemas.Detection` models whose ``bbox`` is
``[left, top, right, bottom]`` in pixels of the saved aerial image, and saves a
copy of that image with the boxes drawn on it.

NOT YET LIVE: ``elements/detect`` answers ``Parameter 'images' is mandatory``.
Fix ``_detect_payload`` from ``notes/vexcel/vexcel_quickstart.py`` (local only)
before relying on it; parsing and drawing are tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel
from shapely.geometry import box

from backend import config
from backend.schemas import Detection
from backend.tools import vexcel

DETECT_ENDPOINT = "elements/detect"


class DetectionRun(BaseModel):
    """Detections for one building image, plus the annotated image path."""

    property_id: int
    prompts: list[str]
    counts: dict[str, int]
    detections: list[Detection]
    image_path: str
    annotated_image_path: str


def _detect_payload(wkt: str, prompts: list[str]) -> dict[str, Any]:
    """Build the detection request body (incomplete: needs 'images')."""
    return {"wkt": wkt, "srid": 4326, "layer": vexcel.LAYERS, "prompts": prompts}


def _lonlat_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return (min_lon, min_lat, max_lon, max_lat) for a GeoJSON-like feature."""
    if isinstance(feature.get("bbox"), list) and len(feature["bbox"]) == 4:
        return tuple(float(v) for v in feature["bbox"])  # type: ignore[return-value]
    geom = feature.get("geometry")
    if not isinstance(geom, dict):
        return None
    coords: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)) and len(node) >= 2 and all(
            isinstance(v, (int, float)) for v in node[:2]
        ):
            coords.append((float(node[0]), float(node[1])))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(geom.get("coordinates"))
    if not coords:
        return None
    xs, ys = zip(*coords)
    return (min(xs), min(ys), max(xs), max(ys))


def parse_detections(
    data: dict[str, Any],
    prompts: list[str],
    property_id: int,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> list[Detection]:
    """Turn a detection response into Detections with pixel boxes.

    *bounds* is the image's lon/lat extent and *image_size* its (width, height);
    north is up. A single-prompt request labels unlabelled features with it.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    width, height = image_size
    features = data.get("features") or data.get("detections") or []
    out: list[Detection] = []
    for feat in features if isinstance(features, list) else []:
        if not isinstance(feat, dict) or (ll := _lonlat_bbox(feat)) is None:
            continue
        props = feat.get("properties") or {}
        label = (props.get("label") or props.get("prompt") or props.get("class")
                 or (prompts[0] if len(prompts) == 1 else "object"))
        score = props.get("score", props.get("confidence"))
        left = (ll[0] - min_lon) / (max_lon - min_lon) * width
        right = (ll[2] - min_lon) / (max_lon - min_lon) * width
        top = (max_lat - ll[3]) / (max_lat - min_lat) * height
        bottom = (max_lat - ll[1]) / (max_lat - min_lat) * height
        out.append(Detection(
            id=f"det-{property_id}-{len(out) + 1}",
            label=str(label),
            property_id=property_id,
            confidence=float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
            bbox=(round(left, 1), round(top, 1), round(right, 1), round(bottom, 1)),
            geometry_wkt=box(*ll).wkt,
        ))
    return out


def draw_detections(image_path: str | Path, detections: list[Detection], out_path: Path) -> Path:
    """Save a copy of the image with numbered yellow boxes for *detections*."""
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections, start=1):
        if det.bbox is None:
            continue
        left, top, right, bottom = det.bbox
        draw.rectangle([left, top, right, bottom], outline=(255, 214, 0), width=3)
        draw.text((left + 3, top + 2), str(i), fill=(255, 214, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return out_path


def detect(lat: float, lon: float, prompts: list[str], property_id: int) -> DetectionRun:
    """Detect *prompts* (e.g. ``["parking space"]``) in the aerial square around a point.

    Uses the same square as ``vexcel.get_aerial``; both calls are cached.
    """
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("detect() needs at least one prompt")

    bounds = vexcel._square_bounds(lat, lon)
    wkt = box(*vexcel._square_bounds(lat, lon, half_m=vexcel.SQUARE_HALF_M)).wkt
    data = vexcel._client.request_json("POST", DETECT_ENDPOINT, _detect_payload(wkt, prompts))

    aerial = vexcel.get_aerial(lat, lon, property_id)
    image_path = str(aerial.image_path)
    with Image.open(image_path) as img:
        size = img.size
    detections = parse_detections(data, prompts, property_id, bounds, size)
    annotated = draw_detections(
        image_path, detections,
        config.CACHE_DIR / "vexcel" / "annotated" / f"property_{property_id}.jpg",
    )

    counts = {p: 0 for p in prompts}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return DetectionRun(
        property_id=property_id, prompts=prompts, counts=counts, detections=detections,
        image_path=image_path, annotated_image_path=str(annotated),
    )
