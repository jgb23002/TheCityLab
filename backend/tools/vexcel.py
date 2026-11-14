"""Vexcel API wrapper: X-API-KEY auth, longitude-first WKT, SRID 4326.

Every live call is routed through ``backend.cache`` so that
``USE_CACHE_ONLY=true`` never touches the network.  Exponential backoff is
applied on HTTP 429 responses (up to MAX_RETRIES attempts).

Quickstart conventions
----------------------
* Auth header only: ``X-API-KEY: <key>``
* WKT uses **longitude first**: ``POINT(lon lat)``
* SRID = 4326
* Imagery box is built with ``shapely.geometry.box`` (~80 m square)
* Layers: ``urban,urban-r,wide-area,urban-g``
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import Point, box, shape

from backend import config
from backend.cache import CacheMiss, cache_key, load_bytes, load_cached, save_bytes, save_cached

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VEXCEL_NS = "vexcel"
IMAGE_NS = "vexcel/images"
MAX_RETRIES = 5
LAYERS = "urban,urban-r,wide-area,urban-g"
# Approximate metres-to-degrees conversion at mid-latitudes (from quickstart)
_M_PER_DEG_LAT = 111_320.0

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


# ---------------------------------------------------------------------------
# HTTP client with caching and retry
# ---------------------------------------------------------------------------

class _VexcelClient:
    """Cache-first HTTP client with X-API-KEY auth and 429 backoff."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_only: bool | None = None,
    ) -> None:
        self._api_key = api_key or config.VEXCEL_API_KEY
        self._base_url = (base_url or config.VEXCEL_BASE_URL).rstrip("/")
        self._cache_only = cache_only if cache_only is not None else config.USE_CACHE_ONLY

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError("VEXCEL_API_KEY is not set in .env")
        return {"X-API-KEY": self._api_key}

    def _cache_key_for(self, method: str, endpoint: str, payload: dict[str, Any]) -> str:
        return cache_key(VEXCEL_NS, {"method": method, "endpoint": endpoint, **payload})

    def request_json(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Return cached JSON or perform a live request with backoff."""
        ck = self._cache_key_for(method, endpoint, payload)

        cached = load_cached(ck, VEXCEL_NS)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if self._cache_only:
            raise CacheMiss(f"Vexcel cache miss for {endpoint} with key {ck}")

        url = f"{self._base_url}/{endpoint}"
        for attempt in range(MAX_RETRIES + 1):
            if method.upper() == "GET":
                resp = requests.get(url, params=payload, headers=self._headers(), timeout=timeout)
            else:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)

            if resp.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue

            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            save_cached(ck, data, VEXCEL_NS)
            return data

        raise RuntimeError("Vexcel retry limit exceeded")  # pragma: no cover

    def request_bytes(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any],
        timeout: int = 60,
    ) -> bytes:
        """Return cached bytes or perform a live request with backoff."""
        ck = self._cache_key_for(method, endpoint, payload)

        cached_data = load_bytes(ck, IMAGE_NS)
        if cached_data is not None:
            return cached_data

        if self._cache_only:
            raise CacheMiss(f"Vexcel image cache miss for {endpoint} with key {ck}")

        url = f"{self._base_url}/{endpoint}"
        for attempt in range(MAX_RETRIES + 1):
            if method.upper() == "GET":
                resp = requests.get(url, params=payload, headers=self._headers(), timeout=timeout)
            else:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)

            if resp.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue

            resp.raise_for_status()
            save_bytes(ck, resp.content, IMAGE_NS)
            return resp.content

        raise RuntimeError("Vexcel retry limit exceeded")  # pragma: no cover


_client = _VexcelClient()


SQUARE_HALF_M = 40.0  # ~80 m square around the building
BUFFER_M = 10  # context buffer Vexcel adds around the square


def _square_bounds(
    lat: float, lon: float, half_m: float = SQUARE_HALF_M + BUFFER_M,
) -> tuple[float, float, float, float]:
    """Return lon-first (min_lon, min_lat, max_lon, max_lat) of a square.

    The default covers the returned image: the ~80 m square plus the 10 m
    buffer.  Uses the quickstart's metres-to-degrees approximation.
    """
    m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(lat))
    delta_lat = half_m / _M_PER_DEG_LAT
    delta_lon = half_m / m_per_deg_lon
    return (lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def get_property_info(lat: float, lon: float) -> dict[str, JsonValue]:
    """Fetch precomputed Vexcel attributes for the parcel at (lat, lon).

    Uses ``GET property/extract`` with longitude-first ``POINT(lon lat)``
    WKT, ``srid=4326``, ``latest=true``, ``include=precomputed``, and
    ``unit-system=imperial``.  Every call is cached.

    Args:
        lat: Latitude of the building centroid.
        lon: Longitude of the building centroid (negative for western hemisphere).

    Returns:
        JSON-serialisable dict with Vexcel precomputed attribute fields, or an
        empty dict if the API returns no features.
    """
    payload: dict[str, Any] = {
        "wkt": f"POINT({lon} {lat})",
        "srid": 4326,
        "layer": LAYERS,
        "latest": "true",
        "include": "precomputed",
        "unit-system": "imperial",
    }
    data = _client.request_json("GET", "property/extract", payload)

    # Vexcel returns every parcel near the point; keep the one that contains
    # it (else the nearest), so neighbours' attributes never leak in.
    features = data.get("features") or data.get("feature") or []
    best = _parcel_for_point(features, lat, lon) if isinstance(features, list) else None
    if best is not None:
        return {"raw": data, "attributes": best.get("properties") or {}}

    return {"raw": data, "attributes": {}}


def _parcel_for_point(features: list[Any], lat: float, lon: float) -> dict[str, Any] | None:
    """Return the feature whose geometry contains (lon, lat), else the nearest one."""
    point = Point(lon, lat)
    best: dict[str, Any] | None = None
    best_dist = math.inf
    for feat in features:
        if not isinstance(feat, dict):
            continue
        try:
            geom = shape(feat["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError):
            dist = math.inf - 1  # no usable geometry: only chosen if nothing else
        else:
            if geom.contains(point):
                return feat
            dist = geom.distance(point)
        if best is None or dist < best_dist:
            best, best_dist = feat, dist
    return best


def get_aerial(lat: float, lon: float, property_id: int) -> dict[str, JsonValue]:
    """Fetch and cache a top-down aerial JPEG for a building.

    Constructs an ~80 m bounding box around (lat, lon) using ``shapely.box``
    with a metres-to-degrees approximation, then calls
    ``POST ortho/extract`` with ``bounding_box`` crop and JPEG format.

    Args:
        lat: Latitude of the building centroid.
        lon: Longitude of the building centroid.
        property_id: CoStar property id (used for the saved image filename).

    Returns:
        A JSON-serialisable dict with keys:
        - ``image_path``: absolute path to the saved JPEG
        - ``property_id``: the passed-in id
        - ``latitude``, ``longitude``: the centroid
        - ``metadata``: raw Vexcel response headers / info dict (may be empty)
    """
    wkt = box(*_square_bounds(lat, lon, half_m=SQUARE_HALF_M)).wkt  # lon first

    # Same body as scripts/vexcel_test.py, which is verified against the live API.
    payload: dict[str, Any] = {
        "wkt": wkt,
        "srid": 4326,
        "layer": LAYERS,
        "buffer-meter": BUFFER_M,
        "crop": "bounding_box",
        "image-format": "jpeg",
        "drawing": [{"wkt": wkt, "outline-width": 4, "outline-color": "rgb(255,204,0)"}],
    }

    img_bytes = _client.request_bytes("POST", "ortho/extract", payload)

    # Save to a human-readable path as well (for the frontend to serve)
    img_dir = config.CACHE_DIR / "vexcel" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"property_{property_id}.jpg"
    img_path.write_bytes(img_bytes)

    return {
        "image_path": str(img_path),
        "property_id": property_id,
        "latitude": lat,
        "longitude": lon,
        "metadata": {},
    }


def get_precomputed(lat: float, lon: float) -> dict[str, JsonValue]:
    """Team-plan name for :func:`get_property_info`.

    Returns Vexcel's ready-made parcel attributes (tree cover, pool, roof ...)
    as ``{"raw": ..., "attributes": {...}}``.
    """
    return get_property_info(lat, lon)


# ---------------------------------------------------------------------------
# Custom-element detection
# ---------------------------------------------------------------------------

# NOTE: endpoint name and payload follow notes/audit.md ("custom-elements
# elements/detect").  Check them against notes/vexcel/vexcel_quickstart.py
# (local only, not in git) before the first live call; only these two
# constants and _detect_payload should need to change.
DETECT_ENDPOINT = "elements/detect"
DETECT_NS = "vexcel/detect"


def _detect_payload(wkt: str, prompts: list[str]) -> dict[str, Any]:
    """Build the detection request body for one image square."""
    return {
        "wkt": wkt,
        "srid": 4326,
        "layer": LAYERS,
        "prompts": prompts,
    }


def _feature_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return lon/lat (minx, miny, maxx, maxy) for a GeoJSON-like feature."""
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


def parse_detections(data: dict[str, Any], prompts: list[str]) -> list[dict[str, JsonValue]]:
    """Turn a detection response into a flat list of boxes.

    Each box is ``{"label", "score", "bbox_lonlat": [minx, miny, maxx, maxy]}``.
    A single-prompt request labels unlabelled features with that prompt.
    """
    features = data.get("features") or data.get("detections") or []
    boxes: list[dict[str, JsonValue]] = []
    for feat in features if isinstance(features, list) else []:
        if not isinstance(feat, dict):
            continue
        bbox = _feature_bbox(feat)
        if bbox is None:
            continue
        props = feat.get("properties") or {}
        label = (
            props.get("label") or props.get("prompt") or props.get("class")
            or (prompts[0] if len(prompts) == 1 else "object")
        )
        score = props.get("score", props.get("confidence"))
        boxes.append({
            "label": str(label),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "bbox_lonlat": list(bbox),
        })
    return boxes


def _draw_boxes(
    image_path: Path,
    boxes: list[dict[str, JsonValue]],
    bounds: tuple[float, float, float, float],
    out_path: Path,
) -> Path:
    """Draw *boxes* (lon/lat) onto the aerial JPEG and save to *out_path*."""
    from PIL import Image, ImageDraw

    min_lon, min_lat, max_lon, max_lat = bounds
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    def to_px(lon_v: float, lat_v: float) -> tuple[float, float]:
        x = (lon_v - min_lon) / (max_lon - min_lon) * width
        y = (max_lat - lat_v) / (max_lat - min_lat) * height  # north is up
        return x, y

    for i, b in enumerate(boxes, start=1):
        x0, y1 = to_px(b["bbox_lonlat"][0], b["bbox_lonlat"][1])  # type: ignore[index]
        x1, y0 = to_px(b["bbox_lonlat"][2], b["bbox_lonlat"][3])  # type: ignore[index]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 214, 0), width=3)
        draw.text((x0 + 3, y0 + 2), str(i), fill=(255, 214, 0))
        b["bbox_px"] = [round(x0), round(y0), round(x1), round(y1)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return out_path


def detect(
    lat: float,
    lon: float,
    prompts: list[str],
    property_id: int = 0,
) -> dict[str, JsonValue]:
    """Find objects like ``"parking space"`` in the aerial image around a point.

    Calls Vexcel detection for the same ~80 m square as :func:`get_aerial`,
    then draws the returned boxes on that image.  Both calls are cached, so
    ``USE_CACHE_ONLY=true`` replays them without network.

    Args:
        lat: Latitude of the building centroid.
        lon: Longitude of the building centroid.
        prompts: Plain words for what to find, e.g. ``["parking space"]``.
        property_id: CoStar id, used to name the saved images.

    Returns:
        ``{"property_id", "prompts", "counts": {prompt: n}, "boxes": [...],
        "image_path", "annotated_image_path"}``.  Each box has ``label``,
        ``score``, ``bbox_lonlat`` and ``bbox_px`` (pixel box on the image).
    """
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("detect() needs at least one prompt")

    bounds = _square_bounds(lat, lon)
    data = _client.request_json(
        "POST", DETECT_ENDPOINT, _detect_payload(box(*bounds).wkt, prompts)
    )
    boxes = parse_detections(data, prompts)

    aerial = get_aerial(lat, lon, property_id)
    image_path = Path(str(aerial["image_path"]))
    annotated = _draw_boxes(
        image_path,
        boxes,
        bounds,
        config.CACHE_DIR / "vexcel" / "annotated" / f"property_{property_id}.jpg",
    )

    counts = {p: 0 for p in prompts}
    for b in boxes:
        counts[str(b["label"])] = counts.get(str(b["label"]), 0) + 1

    return {
        "property_id": property_id,
        "prompts": prompts,  # type: ignore[dict-item]
        "counts": counts,  # type: ignore[dict-item]
        "boxes": boxes,  # type: ignore[dict-item]
        "image_path": str(image_path),
        "annotated_image_path": str(annotated),
    }
