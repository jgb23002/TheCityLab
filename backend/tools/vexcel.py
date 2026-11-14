"""Vexcel API wrapper: X-API-KEY auth, longitude-first WKT, SRID 4326.

Every live call is cached under ``data/cache/vexcel`` so ``USE_CACHE_ONLY=true``
never touches the network. HTTP 429 responses are retried with exponential
backoff. Request fields match ``scripts/vexcel_test.py``, which was checked
against the live API. Detection lives in ``backend/vexcel_detect.py``.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any

import requests
from shapely.geometry import Point, box, shape

from backend import config
from backend.cache import CacheMiss, cache_key, load_bytes, load_cached, save_bytes, save_cached
from backend.schemas import AerialImage, VexcelPropertyInfo

VEXCEL_NS = "vexcel"
IMAGE_NS = "vexcel/images"
MAX_RETRIES = 5
LAYERS = "urban,urban-r,wide-area,urban-g"
SQUARE_HALF_M = 40.0  # ~80 m square around the building
BUFFER_M = 10  # context buffer Vexcel adds around the square
_M_PER_DEG_LAT = 111_320.0  # quickstart's metres-to-degrees approximation


# ---------------------------------------------------------------------------
# HTTP client with caching and retry
# ---------------------------------------------------------------------------

class _VexcelClient:
    """Cache-first HTTP client with X-API-KEY auth and 429 backoff."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or config.VEXCEL_API_KEY
        self._base_url = (base_url or config.VEXCEL_BASE_URL).rstrip("/")

    def _send(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int) -> requests.Response:
        """Send one request, retrying on 429; raise for other HTTP errors."""
        if not self._api_key:
            raise RuntimeError("VEXCEL_API_KEY is not set in .env")
        url = f"{self._base_url}/{endpoint}"
        headers = {"X-API-KEY": self._api_key}
        for attempt in range(MAX_RETRIES + 1):
            if method.upper() == "GET":
                resp = requests.get(url, params=payload, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue
            resp.raise_for_status()  # errors are never cached
            return resp
        raise RuntimeError("Vexcel retry limit exceeded")  # pragma: no cover

    def request_json(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
        """Return cached JSON or perform a live request."""
        key = cache_key(VEXCEL_NS, {"method": method, "endpoint": endpoint, **payload})
        cached = load_cached(key, VEXCEL_NS)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if config.USE_CACHE_ONLY:  # pragma: no cover - load_cached raises first
            raise CacheMiss(f"Vexcel cache miss for {endpoint}")
        data: dict[str, Any] = self._send(method, endpoint, payload, timeout).json()
        save_cached(key, data, VEXCEL_NS)
        return data

    def request_bytes(self, method: str, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> bytes:
        """Return cached bytes or perform a live request."""
        key = cache_key(VEXCEL_NS, {"method": method, "endpoint": endpoint, **payload})
        cached = load_bytes(key, IMAGE_NS)
        if cached is not None:
            return cached
        content = self._send(method, endpoint, payload, timeout).content
        save_bytes(key, content, IMAGE_NS)
        return content


_client = _VexcelClient()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _square_bounds(
    lat: float, lon: float, half_m: float = SQUARE_HALF_M + BUFFER_M,
) -> tuple[float, float, float, float]:
    """Return lon-first (min_lon, min_lat, max_lon, max_lat) of a square.

    The default covers the returned image: the ~80 m square plus the buffer.
    """
    delta_lat = half_m / _M_PER_DEG_LAT
    delta_lon = half_m / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return (lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)


def _parcel_for_point(features: list[Any], lat: float, lon: float) -> dict[str, Any] | None:
    """Return the feature whose geometry contains (lon, lat), else the nearest one."""
    point = Point(lon, lat)
    best: dict[str, Any] | None = None
    best_dist = math.inf
    for feat in features:
        if not isinstance(feat, dict):
            continue
        try:
            dist = 0.0 if (geom := shape(feat["geometry"])).contains(point) else geom.distance(point)
        except (KeyError, TypeError, ValueError, AttributeError):
            dist = math.inf  # no usable geometry: chosen only if nothing else is
        if dist == 0.0:
            return feat
        if best is None or dist < best_dist:
            best, best_dist = feat, dist
    return best


def _point_id(prefix: str, lat: float, lon: float) -> str:
    """Return a short stable id for a coordinate."""
    return f"{prefix}-" + hashlib.sha256(f"{lat:.7f},{lon:.7f}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public tool functions (signatures frozen in docs/CONTRACTS.md)
# ---------------------------------------------------------------------------

def get_aerial(lat: float, lon: float, property_id: int) -> AerialImage:
    """Fetch, cache and save a top-down aerial JPEG of the ~80 m square around a point.

    Calls ``POST ortho/extract`` with a longitude-first square, a 10 m buffer
    and a gold outline. The image is saved to
    ``data/cache/vexcel/images/property_<id>.jpg``.
    """
    wkt = box(*_square_bounds(lat, lon, half_m=SQUARE_HALF_M)).wkt
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

    img_path = config.CACHE_DIR / "vexcel" / "images" / f"property_{property_id}.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(img_bytes)

    min_lon, min_lat, max_lon, max_lat = _square_bounds(lat, lon)
    return AerialImage(
        id=f"aerial-{property_id}",
        property_id=property_id,
        image_path=str(img_path),
        latitude=lat,
        longitude=lon,
        metadata={
            "bounds_lonlat": [min_lon, min_lat, max_lon, max_lat],
            "square_m": SQUARE_HALF_M * 2,
            "buffer_m": BUFFER_M,
            "source": "Vexcel ortho/extract",
        },
    )


def get_property_info(lat: float, lon: float) -> VexcelPropertyInfo:
    """Fetch Vexcel's precomputed attributes for the parcel containing a point.

    Calls ``GET property/extract`` (``POINT(lon lat)``, SRID 4326,
    ``include=precomputed``, imperial units). Vexcel returns every nearby
    parcel; the one containing the point (else the nearest) is used, so a
    neighbour's tree cover never leaks in. Provider keys stay unchanged.
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
    features = data.get("features") or []
    parcel = _parcel_for_point(features, lat, lon) if isinstance(features, list) else None

    props = dict((parcel or {}).get("properties") or {})
    structures = props.pop("structures", None)
    return VexcelPropertyInfo(
        id=_point_id("vexcel", lat, lon),
        latitude=lat,
        longitude=lon,
        attributes=props,
        structures=structures if isinstance(structures, list) else None,
        raw_response=data,
    )
