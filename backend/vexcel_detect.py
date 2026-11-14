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

import math
from typing import Any

import base64
import io

from PIL import Image, ImageDraw
from pydantic import BaseModel
from shapely import wkt as shapely_wkt
from shapely.geometry import box, shape

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


def _image_request(
    area_wkt: str, collection: str | None = None, layer: str | None = None,
) -> dict[str, Any]:
    """The ortho request detection runs on; the downloaded picture must match it.

    With *collection* (and its *layer*), a specific capture date is used.
    """
    req: dict[str, Any] = {"layer": layer or vexcel.LAYERS, "srid": 4326,
                           "wkt": area_wkt, "crop": "bounding_box"}
    if collection:
        req["collection"] = collection
    return req


def _detect_payload(
    area_wkt: str, prompts: list[str], include: tuple[str, ...] = ("bbox", "confidence"),
    image_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the elements/detect body exactly as the quickstart does."""
    return {
        "wkt": area_wkt,
        "srid": 4326,
        "buffer-meter": 0,  # only the area itself, no neighbours
        "images": [{"endpoint": "ortho/extract",
                    "request": image_request or _image_request(area_wkt)}],
        "model-input": {
            "elements": [
                {"label": p, "model": MODEL, "detection-threshold": DETECTION_THRESHOLD}
                for p in prompts
            ],
            "include": list(include),  # the API needs pixel-bbox with pixel-mask
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


def detect(
    lat: float, lon: float, prompts: list[str], property_id: int,
    half_m: float = vexcel.SQUARE_HALF_M,
) -> DetectionRun:
    """Detect *prompts* (e.g. ``["parking space"]``) in a square around a point.

    The square is ``2 * half_m`` metres across (default ~80 m). For a whole
    property, size it from CoStar ``lotSize``: half_m = sqrt(acres * 4047) / 2.

    Returns counts per prompt, the Detections, the clean image path and an
    annotated copy with numbered boxes.
    """
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("detect() needs at least one prompt")

    bounds = vexcel._square_bounds(lat, lon, half_m=half_m)
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
    tag = "" if half_m == vexcel.SQUARE_HALF_M else f"_{round(2 * half_m)}m"
    image_path = img_dir / f"property_{property_id}{tag}.jpg"
    image_path.write_bytes(img_bytes)
    with Image.open(image_path) as img:
        size = img.size

    detections = parse_detections(data, property_id, bounds, size)
    annotated = draw_detections(
        str(image_path), detections, img_dir / f"property_{property_id}{tag}_boxes.jpg"
    )

    counts = {p: 0 for p in prompts}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return DetectionRun(
        property_id=property_id, prompts=prompts, counts=counts, detections=detections,
        image_path=str(image_path), annotated_image_path=annotated,
    )


def _dedupe(detections: list[Detection], iou_min: float = 0.3) -> list[Detection]:
    """Drop a detection that overlaps an earlier, more confident one of the same label.

    Objects on a tile edge can be found by both neighbouring tiles.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -(d.confidence or 0)):
        g = shapely_wkt.loads(det.geometry_wkt) if det.geometry_wkt else None
        dup = False
        for k in kept:
            if g is None or k.label != det.label or not k.geometry_wkt:
                continue
            other = shapely_wkt.loads(k.geometry_wkt)
            union = g.union(other).area
            if union and g.intersection(other).area / union >= iou_min:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


def detect_tiled(
    lat: float, lon: float, prompts: list[str], property_id: int,
    half_m: float, tile_m: float = 80.0,
) -> DetectionRun:
    """Detect *prompts* over a large square by searching ~80 m tiles one by one.

    The detector finds small things (single empty spaces) on an ~80 m image
    but misses them on one big image, so the square is split into an n x n
    grid of tiles of at most *tile_m* metres. Each tile is one cached
    elements/detect call; boxes are merged, de-duplicated across tile edges
    and drawn on the single large ortho image.
    """
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("detect_tiled() needs at least one prompt")

    bounds = vexcel._square_bounds(lat, lon, half_m=half_m)
    min_lon, min_lat, max_lon, max_lat = bounds
    n = max(1, math.ceil(2 * half_m / tile_m))
    step_lon, step_lat = (max_lon - min_lon) / n, (max_lat - min_lat) / n

    raw: list[Detection] = []
    for row in range(n):
        for col in range(n):
            tile = (min_lon + col * step_lon, min_lat + row * step_lat,
                    min_lon + (col + 1) * step_lon, min_lat + (row + 1) * step_lat)
            tile_wkt = box(*tile).wkt
            data = vexcel._client.request_json(
                "POST", DETECT_ENDPOINT, _detect_payload(tile_wkt, prompts), timeout=300
            )
            # Pixel boxes are recomputed on the big image below from geometry_wkt.
            raw.extend(parse_detections(data, property_id, bounds, (1, 1)))

    area_wkt = box(*bounds).wkt
    img_bytes = vexcel._client.request_bytes(
        "POST", "ortho/extract",
        {**_image_request(area_wkt), "buffer-meter": 0, "image-format": "jpeg"},
    )
    img_dir = config.CACHE_DIR / "vexcel" / "detect"
    img_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{round(2 * half_m)}m_tiled"
    image_path = img_dir / f"property_{property_id}{tag}.jpg"
    image_path.write_bytes(img_bytes)
    with Image.open(image_path) as img:
        width, height = img.size

    detections: list[Detection] = []
    for i, det in enumerate(_dedupe(raw), start=1):
        x0, y0, x1, y1 = shapely_wkt.loads(det.geometry_wkt).bounds  # type: ignore[arg-type]
        detections.append(det.model_copy(update={
            "id": f"det-{property_id}-{i}",
            "bbox": (
                round((x0 - min_lon) / (max_lon - min_lon) * width, 1),
                round((max_lat - y1) / (max_lat - min_lat) * height, 1),
                round((x1 - min_lon) / (max_lon - min_lon) * width, 1),
                round((max_lat - y0) / (max_lat - min_lat) * height, 1),
            ),
        }))
    annotated = draw_detections(str(image_path), detections, img_dir / f"property_{property_id}{tag}_boxes.jpg")

    counts = {p: 0 for p in prompts}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return DetectionRun(
        property_id=property_id, prompts=prompts, counts=counts, detections=detections,
        image_path=str(image_path), annotated_image_path=annotated,
    )


# ---------------------------------------------------------------------------
# Parking estimate from paved lot area (method B)
# ---------------------------------------------------------------------------

SQFT_PER_M2 = 10.7639
# One space plus its share of the drive aisle (US rule of thumb).
SQFT_PER_SPACE = (300.0, 350.0)


class ParkingEstimate(BaseModel):
    """Spaces estimated as measured parking-lot area / area per space."""

    property_id: int
    lot_count: int
    lot_area_sq_ft: float
    spaces_low: int
    spaces_high: int
    spaces_mid: int
    sq_ft_per_space: tuple[float, float]
    tiles: int
    detect_calls: int
    annotated_image_path: str
    method: str


def _mask_fraction(mask_png_b64: str) -> float | None:
    """Share of the mask picture that is object (white), 0..1."""
    try:
        mask = Image.open(io.BytesIO(base64.b64decode(mask_png_b64))).convert("L")
    except Exception:
        return None
    hist = mask.histogram()
    total = sum(hist)
    return sum(hist[129:]) / total if total else None


def _box_area_m2(geom: Any) -> float:
    """Area in m^2 of a lon/lat box, using the local metres-per-degree scale."""
    x0, y0, x1, y1 = geom.bounds
    lat = (y0 + y1) / 2
    return ((x1 - x0) * 111_320.0 * math.cos(math.radians(lat))) * ((y1 - y0) * 111_320.0)


def estimate_parking_spaces(
    lat: float, lon: float, property_id: int, half_m: float,
    tile_m: float = 80.0, label: str = "parking lot",
) -> ParkingEstimate:
    """Estimate surface parking spaces from detected parking-lot area.

    The square is tiled like :func:`detect_tiled`; each tile asks for
    *label* with pixel masks. Lot area = box area x share of the box the mask
    covers. Spaces = area / 350 (low) to area / 300 (high) sq ft per space.
    Lots cut by a tile edge are measured piece by piece, so nothing is
    counted twice; overlapping duplicates are dropped first.
    """
    bounds = vexcel._square_bounds(lat, lon, half_m=half_m)
    min_lon, min_lat, max_lon, max_lat = bounds
    n = max(1, math.ceil(2 * half_m / tile_m))
    step_lon, step_lat = (max_lon - min_lon) / n, (max_lat - min_lat) / n
    include = ("bbox", "confidence", "pixel-bbox", "pixel-mask")

    lots: list[tuple[Detection, float]] = []  # (detection, area m^2)
    for row in range(n):
        for col in range(n):
            tile = (min_lon + col * step_lon, min_lat + row * step_lat,
                    min_lon + (col + 1) * step_lon, min_lat + (row + 1) * step_lat)
            data = vexcel._client.request_json(
                "POST", DETECT_ENDPOINT, _detect_payload(box(*tile).wkt, [label], include),
                timeout=300,
            )
            images = data.get("images") or []
            ortho_id = images[0].get("image-id") if images else None
            for feat in data.get("features") or []:
                for det in ((feat or {}).get("properties") or {}).get("detections") or []:
                    if ortho_id is not None and det.get("imageId") != ortho_id:
                        continue
                    try:
                        geom = shapely_wkt.loads(det["bbox"]).intersection(box(*tile))
                    except Exception:
                        continue
                    if geom.is_empty:
                        continue
                    frac = _mask_fraction(det["pixel-mask"]) if det.get("pixel-mask") else None
                    conf = det.get("confidence")
                    lots.append((
                        Detection(
                            id=f"lot-{property_id}-{len(lots) + 1}", label=label,
                            property_id=property_id, geometry_wkt=geom.wkt,
                            confidence=float(conf) if isinstance(conf, (int, float)) and 0 <= conf <= 1 else None,
                        ),
                        _box_area_m2(geom) * (frac if frac is not None else 1.0),
                    ))

    area_by_id = {d.id: a for d, a in lots}
    kept = _dedupe([d for d, _ in lots], iou_min=0.5)
    area_sqft = sum(area_by_id[d.id] for d in kept) * SQFT_PER_M2

    img_bytes = vexcel._client.request_bytes(
        "POST", "ortho/extract",
        {**_image_request(box(*bounds).wkt), "buffer-meter": 0, "image-format": "jpeg"},
    )
    img_dir = config.CACHE_DIR / "vexcel" / "detect"
    img_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{round(2 * half_m)}m_lots"
    image_path = img_dir / f"property_{property_id}{tag}.jpg"
    image_path.write_bytes(img_bytes)
    with Image.open(image_path) as img:
        width, height = img.size
    drawn = []
    for d in kept:
        x0, y0, x1, y1 = shapely_wkt.loads(d.geometry_wkt).bounds  # type: ignore[arg-type]
        drawn.append(d.model_copy(update={"bbox": (
            (x0 - min_lon) / (max_lon - min_lon) * width, (max_lat - y1) / (max_lat - min_lat) * height,
            (x1 - min_lon) / (max_lon - min_lon) * width, (max_lat - y0) / (max_lat - min_lat) * height,
        )}))
    annotated = draw_detections(str(image_path), drawn, img_dir / f"property_{property_id}{tag}_boxes.jpg")

    low, high = SQFT_PER_SPACE
    return ParkingEstimate(
        property_id=property_id,
        lot_count=len(kept),
        lot_area_sq_ft=round(area_sqft),
        spaces_low=round(area_sqft / high),
        spaces_high=round(area_sqft / low),
        spaces_mid=round(area_sqft / ((low + high) / 2)),
        sq_ft_per_space=SQFT_PER_SPACE,
        tiles=n * n,
        detect_calls=n * n,
        annotated_image_path=annotated,
        method=f"'{label}' masks on {n}x{n} tiles; area / {low:g}-{high:g} sq ft per space",
    )


# ---------------------------------------------------------------------------
# Spaces from cars seen on several capture dates (method: union over time)
# ---------------------------------------------------------------------------

SHARP_LAYERS = ("urban", "urban-r")  # 7.5 cm orthos
SAME_SPOT_M = 2.0  # same spot across dates (orthos shift ~1-2 m; spaces are 2.7 m apart)


class DateCount(BaseModel):
    """Cars found on one capture date."""

    collection: str
    layer: str
    date: str
    cars: int


class MultiDateSpaces(BaseModel):
    """Distinct parking spots = car positions merged across capture dates."""

    property_id: int
    dates: list[DateCount]
    spaces: int
    spaces_seen_once: int
    spaces_two_plus_dates: int = 0
    empty_spots_added: int = 0
    estimate: int = 0
    spots: list[dict[str, Any]] = []  # {lon, lat, dates, empty_only}
    detect_calls: int
    annotated_image_path: str
    method: str


def list_collections(lat: float, lon: float) -> list[dict[str, str]]:
    """Ortho capture dates at a point, newest first: {collection, layer, date}."""
    data = vexcel._client.request_json("GET", "ortho/collections", {
        "wkt": f"POINT({lon} {lat})", "srid": 4326, "layer": vexcel.LAYERS,
        "include": "collection,last-capture-date,layer",
    })
    out = []
    for f in data.get("features") or []:
        p = (f or {}).get("properties") or {}
        if p.get("collection") and p.get("last-capture-date"):
            out.append({"collection": p["collection"], "layer": p.get("layer", ""),
                        "date": p["last-capture-date"][:10]})
    return sorted(out, key=lambda c: c["date"], reverse=True)


def _centre_m(geom_wkt: str, lat0: float) -> tuple[float, float]:
    """Box centre in metres (east, north) on a local flat grid."""
    c = shapely_wkt.loads(geom_wkt).centroid
    return (c.x * 111_320.0 * math.cos(math.radians(lat0)), c.y * 111_320.0)


def property_outline(
    lat: float, lon: float, lot_acres: float, search_half_m: float | None = None,
) -> str | None:
    """WKT of the parcel that best matches the CoStar property, or None.

    CoStar's point may sit off-centre or inside a small neighbouring parcel,
    so this searches all Vexcel parcels in a square around the point and picks
    the one whose area is closest to CoStar ``lotSize`` (within 25%). The
    square defaults to the lot's own size: Vexcel returns at most 500 parcels,
    so a much larger square can cut the answer off before the right parcel.
    """
    if search_half_m is None:
        search_half_m = max(100.0, math.sqrt(lot_acres * 4046.86) / 2)
    area = box(*vexcel._square_bounds(lat, lon, half_m=search_half_m)).wkt
    data = vexcel._client.request_json("GET", "property/extract", {
        "wkt": area, "srid": 4326, "layer": vexcel.LAYERS, "latest": "true",
        "include": "precomputed", "unit-system": "imperial",
    })
    target_m2 = lot_acres * 4046.86
    best, best_err = None, 0.25
    for f in data.get("features") or []:
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        err = abs(_box_area_m2_poly(g, lat) - target_m2) / target_m2
        if err < best_err:
            best, best_err = g, err
    return best.wkt if best is not None else None


def _box_area_m2_poly(geom: Any, lat: float) -> float:
    """Polygon area in m^2 using the local metres-per-degree scale."""
    return geom.area * 111_320.0 ** 2 * math.cos(math.radians(lat))


def spaces_from_dates(
    lat: float, lon: float, property_id: int, half_m: float = 0.0,
    n_dates: int = 4, tile_m: float = 80.0, label: str = "car",
    outline_wkt: str | None = None, empty_label: str | None = "empty parking spot",
) -> MultiDateSpaces:
    """Estimate parking spaces as distinct spots where a car was parked on any date.

    Runs tiled *label* detection on the *n_dates* newest sharp (7.5 cm)
    captures, then merges car centres closer than ``SAME_SPOT_M`` across dates
    (greedy, most confident first). Each date sees a different set of parked
    cars, so the union approaches the number of spaces.

    With *outline_wkt* (see :func:`property_outline`) the tiles cover that
    outline and only spots inside it (plus 3 m) count, which drops traffic on
    public roads and neighbours' cars. Otherwise a square of *half_m* is used.
    """
    outline = shapely_wkt.loads(outline_wkt) if outline_wkt else None
    if outline is None and half_m <= 0:
        raise ValueError("spaces_from_dates() needs outline_wkt or a positive half_m")
    dates = [c for c in list_collections(lat, lon) if c["layer"] in SHARP_LAYERS][:n_dates]
    if outline is not None:
        bounds = outline.bounds
        keep_area = outline.buffer(3.0 / 111_320.0)
    else:
        bounds = vexcel._square_bounds(lat, lon, half_m=half_m)
        keep_area = box(*bounds)
    min_lon, min_lat, max_lon, max_lat = bounds
    width_m = (max_lon - min_lon) * 111_320.0 * math.cos(math.radians(lat))
    height_m = (max_lat - min_lat) * 111_320.0
    nx, ny = max(1, math.ceil(width_m / tile_m)), max(1, math.ceil(height_m / tile_m))
    step_lon, step_lat = (max_lon - min_lon) / nx, (max_lat - min_lat) / ny

    per_date: list[DateCount] = []
    spots: list[dict[str, Any]] = []  # {xy, det, dates}
    calls = 0
    for c in dates:
        found: list[Detection] = []
        for row in range(ny):
            for col in range(nx):
                tile_geom = box(min_lon + col * step_lon, min_lat + row * step_lat,
                                min_lon + (col + 1) * step_lon, min_lat + (row + 1) * step_lat)
                if not tile_geom.intersects(keep_area):
                    continue
                tile = tile_geom.wkt
                data = vexcel._client.request_json(
                    "POST", DETECT_ENDPOINT,
                    _detect_payload(tile, [label], image_request=_image_request(tile, c["collection"], c["layer"])),
                    timeout=300,
                )
                calls += 1
                found.extend(parse_detections(data, property_id, bounds, (1, 1)))
        found = [d for d in _dedupe(found)
                 if keep_area.contains(shapely_wkt.loads(d.geometry_wkt).centroid)]  # type: ignore[arg-type]
        per_date.append(DateCount(collection=c["collection"], layer=c["layer"], date=c["date"], cars=len(found)))

        for det in sorted(found, key=lambda d: -(d.confidence or 0)):
            xy = _centre_m(det.geometry_wkt, lat)  # type: ignore[arg-type]
            match = next((s for s in spots if math.dist(s["xy"], xy) < SAME_SPOT_M), None)
            if match:
                match["dates"].add(c["date"])
            else:
                spots.append({"xy": xy, "det": det, "dates": {c["date"]}})

    # Combined estimate: a spot reused on 2+ dates is a real space (a car
    # stopped in a drive lane rarely stops in the same place twice); plus
    # empty spaces from the newest photo that no car ever covered.
    confirmed = [s for s in spots if len(s["dates"]) >= 2]
    empty_added = 0
    if empty_label and dates:
        c = dates[0]
        empties: list[Detection] = []
        for row in range(ny):
            for col in range(nx):
                tile_geom = box(min_lon + col * step_lon, min_lat + row * step_lat,
                                min_lon + (col + 1) * step_lon, min_lat + (row + 1) * step_lat)
                if not tile_geom.intersects(keep_area):
                    continue
                data = vexcel._client.request_json(
                    "POST", DETECT_ENDPOINT,
                    _detect_payload(tile_geom.wkt, [empty_label],
                                    image_request=_image_request(tile_geom.wkt, c["collection"], c["layer"])),
                    timeout=300,
                )
                calls += 1
                empties.extend(parse_detections(data, property_id, bounds, (1, 1)))
        for det in _dedupe(empties):
            if not keep_area.contains(shapely_wkt.loads(det.geometry_wkt).centroid):  # type: ignore[arg-type]
                continue
            xy = _centre_m(det.geometry_wkt, lat)  # type: ignore[arg-type]
            if any(math.dist(s["xy"], xy) < SAME_SPOT_M for s in spots):
                continue
            spots.append({"xy": xy, "det": det, "dates": {"empty:" + c["date"]}})
            confirmed.append(spots[-1])
            empty_added += 1

    # Draw every distinct spot on the newest image.
    newest = dates[0] if dates else {"collection": None, "layer": None}
    img_bytes = vexcel._client.request_bytes("POST", "ortho/extract", {
        **_image_request(box(*bounds).wkt, newest["collection"], newest["layer"]),
        "buffer-meter": 0, "image-format": "jpeg",
    })
    img_dir = config.CACHE_DIR / "vexcel" / "detect"
    img_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_outline_dates{len(dates)}" if outline is not None else f"_{round(2 * half_m)}m_dates{len(dates)}"
    image_path = img_dir / f"property_{property_id}{tag}.jpg"
    image_path.write_bytes(img_bytes)
    with Image.open(image_path) as img:
        width, height = img.size
    drawn = []
    for i, s in enumerate(spots, start=1):
        x0, y0, x1, y1 = shapely_wkt.loads(s["det"].geometry_wkt).bounds
        drawn.append(s["det"].model_copy(update={"id": f"spot-{property_id}-{i}", "bbox": (
            (x0 - min_lon) / (max_lon - min_lon) * width, (max_lat - y1) / (max_lat - min_lat) * height,
            (x1 - min_lon) / (max_lon - min_lon) * width, (max_lat - y0) / (max_lat - min_lat) * height,
        )}))
    annotated = draw_detections(str(image_path), drawn, img_dir / f"property_{property_id}{tag}_boxes.jpg")

    return MultiDateSpaces(
        property_id=property_id,
        dates=per_date,
        spaces=len(spots),
        spaces_seen_once=sum(1 for s in spots if len(s["dates"]) == 1 and not
                             next(iter(s["dates"])).startswith("empty:")),
        spaces_two_plus_dates=sum(1 for s in spots if len(s["dates"]) >= 2),
        empty_spots_added=empty_added,
        estimate=len(confirmed),
        spots=[{
            "lon": shapely_wkt.loads(sp["det"].geometry_wkt).centroid.x,
            "lat": shapely_wkt.loads(sp["det"].geometry_wkt).centroid.y,
            "dates": len([d for d in sp["dates"] if not d.startswith("empty:")]),
            "empty_only": all(d.startswith("empty:") for d in sp["dates"]),
        } for sp in spots],
        detect_calls=calls,
        annotated_image_path=annotated,
        method=(f"'{label}' on {nx}x{ny} tiles x {len(dates)} dates; spots < {SAME_SPOT_M} m apart merged"
                + ("; inside property outline" if outline is not None else "")),
    )
