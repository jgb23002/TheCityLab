"""Vexcel custom-element detection: find things like "parking space" in an image.

A data helper for ``backend.tools.vision`` (not an agent tool itself). Follows
step 3 of ``notes/vexcel/vexcel_quickstart.py``: ``POST elements/detect`` gets
the area, the list of images to search (each an ordinary ``ortho/extract``
request) and the labels. Each detection comes back as a map-coordinate WKT
box tagged with the image it was found in.

The same ortho request (buffer 0) is downloaded so boxes line up with the
picture; boxes are converted to ``[left, top, right, bottom]`` pixels and drawn
on a copy. Every call is cached, so ``USE_CACHE_ONLY=true`` replays offline.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from backend import config
from backend.schemas import Detection
from backend.tools import vexcel

DETECT_ENDPOINT = "elements/detect"
MODEL = "sam3"
DETECTION_THRESHOLD = 0.5


class DetectionRun(BaseModel):
    """Detections for one building image, plus the image paths."""

    property_id: int
    prompts: list[str]
    counts: dict[str, int]
    detections: list[Detection]
    image_path: str
    annotated_image_path: str


def _image_request(area_wkt: str) -> dict[str, Any]:
    """The ortho request detection runs on; the downloaded picture must match it."""
    return {"layer": vexcel.LAYERS, "srid": 4326, "wkt": area_wkt, "crop": "bounding_box"}


def _detect_payload(area_wkt: str, prompts: list[str]) -> dict[str, Any]:
    """Build the elements/detect body exactly as the quickstart does."""
    return {
        "wkt": area_wkt,
        "srid": 4326,
        "buffer-meter": 0,  # only the area itself, no neighbours
        "images": [{"endpoint": "ortho/extract", "request": _image_request(area_wkt)}],
        "model-input": {
            "elements": [
                {"label": p, "model": MODEL, "detection-threshold": DETECTION_THRESHOLD}
                for p in prompts
            ],
            "include": ["bbox", "confidence"],
        },
    }


def parse_detections(
    data: dict[str, Any],
    property_id: int,
    bounds: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> list[Detection]:
    """Turn an elements/detect response into Detections with pixel boxes.

    Response shape (quickstart): ``images[].image-id`` and
    ``features[].properties = {label, detections: [{imageId, confidence, bbox}]}``
    where ``bbox`` is a WKT polygon in lon/lat. Only detections on the first
    (ortho) image are kept. *bounds* is that image's lon/lat extent and
    *image_size* its (width, height); north is up.
    """
    images = data.get("images") or []
    ortho_id = images[0].get("image-id") if images and isinstance(images[0], dict) else None
    min_lon, min_lat, max_lon, max_lat = bounds
    width, height = image_size

    out: list[Detection] = []
    for feat in data.get("features") or []:
        props = (feat or {}).get("properties") or {}
        label = str(props.get("label") or "object")
        for det in props.get("detections") or []:
            if ortho_id is not None and det.get("imageId") != ortho_id:
                continue
            try:
                geom = shapely_wkt.loads(det["bbox"])
            except Exception:  # missing or malformed WKT (shapely raises GEOSException)
                continue
            x0, y0, x1, y1 = geom.bounds  # lon/lat
            left = (x0 - min_lon) / (max_lon - min_lon) * width
            right = (x1 - min_lon) / (max_lon - min_lon) * width
            top = (max_lat - y1) / (max_lat - min_lat) * height
            bottom = (max_lat - y0) / (max_lat - min_lat) * height
            conf = det.get("confidence")
            out.append(Detection(
                id=f"det-{property_id}-{len(out) + 1}",
                label=label,
                property_id=property_id,
                confidence=float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None,
                bbox=(round(left, 1), round(top, 1), round(right, 1), round(bottom, 1)),
                geometry_wkt=geom.wkt,
            ))
    return out


def draw_detections(image_path: str, detections: list[Detection], out_path: Any) -> str:
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
    return str(out_path)


def detect(lat: float, lon: float, prompts: list[str], property_id: int) -> DetectionRun:
    """Detect *prompts* (e.g. ``["parking space"]``) in the ~80 m square around a point.

    Returns counts per prompt, the Detections, the clean image path and an
    annotated copy with numbered boxes.
    """
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("detect() needs at least one prompt")

    bounds = vexcel._square_bounds(lat, lon, half_m=vexcel.SQUARE_HALF_M)
    area_wkt = box(*bounds).wkt
    data = vexcel._client.request_json(
        "POST", DETECT_ENDPOINT, _detect_payload(area_wkt, prompts), timeout=300
    )

    # Same source image as the detection, so boxes are not shifted.
    img_bytes = vexcel._client.request_bytes(
        "POST", "ortho/extract",
        {**_image_request(area_wkt), "buffer-meter": 0, "image-format": "jpeg"},
    )
    img_dir = config.CACHE_DIR / "vexcel" / "detect"
    img_dir.mkdir(parents=True, exist_ok=True)
    image_path = img_dir / f"property_{property_id}.jpg"
    image_path.write_bytes(img_bytes)
    with Image.open(image_path) as img:
        size = img.size

    detections = parse_detections(data, property_id, bounds, size)
    annotated = draw_detections(
        str(image_path), detections, img_dir / f"property_{property_id}_boxes.jpg"
    )

    counts = {p: 0 for p in prompts}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return DetectionRun(
        property_id=property_id, prompts=prompts, counts=counts, detections=detections,
        image_path=str(image_path), annotated_image_path=annotated,
    )
