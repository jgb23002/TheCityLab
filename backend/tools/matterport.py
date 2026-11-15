"""Optional Matterport GraphQL integration using HTTP Basic authentication.

``get_model`` returns a scan's measured spatial metadata: floors and rooms with their
measured dimensions. Matterport reports two dimension sets; on these models
``dimensions.height`` is usually empty while ``dimensionEstimates.height`` carries real
per-room heights, so heights read the estimate first, whereas floor areas read the
measured ``dimensions.areaFloor`` first and fall back to the estimate.
Room-level heights are a real measurement, not a vision model's photo reading, so
``building_height_ft`` may be preferred over the CoStar-math fallback in
``backend.tools.geometry``. The Model API exposes no interior images, so lighting,
window and HVAC observations are never inferred from it.

Every response is cached under data/cache/matterport/ and USE_CACHE_ONLY is honoured.
"""

from __future__ import annotations

import csv
import statistics
import zipfile
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import requests
from shapely.geometry import MultiPoint

from backend import config
from backend.cache import CacheMiss, cache_key, load_cached, save_cached
from backend.schemas import MatterpakGeometry, MatterportModel

NAMESPACE = "matterport"
# Bump the version whenever the query changes so stale cache entries are not replayed.
_QUERY_VERSION = "spatial-v2"
_MODEL_QUERY = """query($id: ID!) {
  model(id: $id) {
    id name created visibility
    floors {
      id label sequence
      dimensions { areaFloor units }
      dimensionEstimates { area units }
    }
    rooms {
      id label tags
      dimensions { height areaFloor width depth units }
      dimensionEstimates { height area areaIndoor width depth lowerElevation units }
    }
  }
}"""
# A room whose ceiling plane could not be resolved reports height 0.0, not a real height.
_MIN_REAL_HEIGHT = 0.5


@lru_cache(maxsize=1)
def _model_ids() -> dict[int, str]:
    """{CoStar PropertyId: MatterportModelId} from the kit's commercial-properties.csv."""
    path = config.COSTAR_DATA_DIR.parent / "commercial-properties.csv"
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mid = (row.get("MatterportModelId") or "").strip()
            pid = (row.get("PropertyId") or "").strip()
            if mid and pid.isdigit():
                out[int(pid)] = mid
    return out


def model_id_for(property_id: int) -> str | None:
    """The Matterport model id linked to a CoStar property, or None when it has no scan."""
    return _model_ids().get(int(property_id))


def _fetch(model_id: str) -> dict[str, Any]:
    """POST the GraphQL query with Basic auth; raises on HTTP or GraphQL errors."""
    if not (config.MP_TOKEN_ID and config.MP_TOKEN_SECRET):
        raise RuntimeError("Set MP_TOKEN_ID and MP_TOKEN_SECRET in .env for Matterport.")
    response = requests.post(config.MATTERPORT_GRAPHQL_URL, timeout=30,
                             json={"query": _MODEL_QUERY, "variables": {"id": model_id}},
                             auth=(config.MP_TOKEN_ID, config.MP_TOKEN_SECRET))
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError("Matterport query failed: " + "; ".join(str(e.get("message", ""))[:120] for e in body["errors"]))
    model = (body.get("data") or {}).get("model")
    if not model:
        raise LookupError(f"Matterport model {model_id} was not found or is not accessible.")
    return model


def _positive(value: Any) -> float | None:
    """A real measured number, or None for null/zero placeholders."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _measured(item: dict[str, Any], estimate_key: str, dimension_key: str,
              estimate_first: bool = True) -> float | None:
    """Read one value from both dimension sets; the first positive number wins."""
    estimate = _positive((item.get("dimensionEstimates") or {}).get(estimate_key))
    dimension = _positive((item.get("dimensions") or {}).get(dimension_key))
    first, second = (estimate, dimension) if estimate_first else (dimension, estimate)
    return first if first is not None else second


def summarize(model: dict[str, Any]) -> dict[str, Any]:
    """Measured facts only: floor and room counts, floor areas, measured ceiling heights."""
    floors = [f for f in model.get("floors") or [] if isinstance(f, dict)]
    rooms = [r for r in model.get("rooms") or [] if isinstance(r, dict)]
    floor_area = [_measured(f, "area", "areaFloor", estimate_first=False) for f in floors]
    heights = [h for h in (_measured(r, "height", "height") for r in rooms)
               if h is not None and h > _MIN_REAL_HEIGHT]
    units = next((((f.get("dimensions") or f.get("dimensionEstimates") or {}).get("units"))
                  for f in floors if f.get("dimensions") or f.get("dimensionEstimates")), None)
    return {
        "name": model.get("name"),
        "created": model.get("created"),
        "floor_count": len(floors),
        "scanned_floor_area": round(sum(a for a in floor_area if a is not None), 1) if floor_area else None,
        "room_count": len(rooms),
        "labelled_rooms": sorted({r.get("label") for r in rooms if r.get("label")}),
        "measured_ceiling_heights": [round(h, 2) for h in heights],
        "units": units,
        "rooms": [{"label": r.get("label") or None, "tags": r.get("tags") or [],
                   "area_floor": _measured(r, "area", "areaFloor", estimate_first=False),
                   "width": _measured(r, "width", "width"),
                   "depth": _measured(r, "depth", "depth"),
                   "height": _measured(r, "height", "height"),
                   "lower_elevation": (r.get("dimensionEstimates") or {}).get("lowerElevation")} for r in rooms],
    }


def get_model(model_id: str) -> MatterportModel:
    """Fetch (or replay from cache) one Matterport model's measured spatial metadata."""
    key = cache_key(NAMESPACE, {"model_id": model_id, "query": _QUERY_VERSION})
    cached = load_cached(key, NAMESPACE)  # raises CacheMiss when USE_CACHE_ONLY and not cached
    if cached is None:
        if config.USE_CACHE_ONLY:  # pragma: no cover - load_cached raises first
            raise CacheMiss(f"Matterport cache miss for {model_id}")
        cached = _fetch(model_id)
        save_cached(key, cached, NAMESPACE)
    return MatterportModel(id=model_id, metadata=summarize(cached), raw_response=None)


def building_height_ft(model: MatterportModel) -> float | None:
    """Median of the rooms' measured ceiling heights, or None when no room resolved one."""
    heights = (model.metadata or {}).get("measured_ceiling_heights") or []
    heights = [h for h in heights if isinstance(h, (int, float)) and h > _MIN_REAL_HEIGHT]
    return statistics.median(heights) if heights else None


# ------------------------------------------------------------------ MatterPak

# A typical apartment scan is ~75 MB; a warehouse can exceed 400 MB. The CDN sends no
# Content-Length, so the cap is enforced while streaming.
MATTERPAK_MAX_MB = 150
_GEOMETRY_VERSION = "xyz-v1"
_HULL_SAMPLE = 50_000  # points used for the footprint rectangle; a full cloud is millions
_FT_PER_M = 3.28084
_SQ_FT_PER_SQ_M = 10.7639
_BUNDLE_QUERY = """query($id: ID!) {
  model(id: $id) { bundle(id: "mp:matterpak") { assets { filename format status url } } }
}"""


def _bundle_path(model_id: str):
    return config.CACHE_DIR / NAMESPACE / "bundles" / f"{model_id}.zip"


def _floorplan_path(model_id: str):
    return config.CACHE_DIR / NAMESPACE / "floorplans" / f"{model_id}.jpg"


def _bundle_url(model_id: str) -> str | None:
    """The signed download URL of the model's MatterPak zip (it expires, so it is never cached)."""
    if not (config.MP_TOKEN_ID and config.MP_TOKEN_SECRET):
        raise RuntimeError("Set MP_TOKEN_ID and MP_TOKEN_SECRET in .env for Matterport.")
    response = requests.post(config.MATTERPORT_GRAPHQL_URL, timeout=30,
                             json={"query": _BUNDLE_QUERY, "variables": {"id": model_id}},
                             auth=(config.MP_TOKEN_ID, config.MP_TOKEN_SECRET))
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError("Matterport bundle query failed: "
                           + "; ".join(str(e.get("message", ""))[:120] for e in body["errors"]))
    assets = ((((body.get("data") or {}).get("model") or {}).get("bundle") or {}).get("assets")) or []
    for asset in assets:
        if asset.get("status") == "available" and asset.get("url"):
            return asset["url"]
    return None


def _download_bundle(url: str, path, max_bytes: int) -> int:
    """Stream the zip to disk, aborting (and cleaning up) once the byte cap is passed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    received = 0
    with requests.get(url, stream=True, timeout=60) as response, partial.open("wb") as handle:
        response.raise_for_status()
        for chunk in response.iter_content(4 * 1024 * 1024):
            received += len(chunk)
            if received > max_bytes:
                handle.close()
                partial.unlink(missing_ok=True)
                raise ValueError(f"bundle exceeds {max_bytes // (1024 * 1024)} MB")
            handle.write(chunk)
    partial.replace(path)
    return received


def geometry_from_xyz(points: np.ndarray) -> dict[str, Any]:
    """Interior height and footprint from an (n, 3) array of x y z metres, Z-up.

    Height is the 1st-99th percentile span of z (stray points above the ceiling or
    below the floor are ignored). The footprint is the smallest rotated rectangle
    around a random sample of the points, so a building that is not axis-aligned
    is not over-measured.
    """
    z = points[:, 2]
    low, high = np.percentile(z, [1, 99])
    sample = points
    if len(points) > _HULL_SAMPLE:
        sample = points[np.random.default_rng(0).choice(len(points), _HULL_SAMPLE, replace=False)]
    footprint_m2 = MultiPoint(sample[:, :2].tolist()).minimum_rotated_rectangle.area
    return {
        "point_count": int(len(points)),
        "interior_height_ft": round(float(high - low) * _FT_PER_M, 2),
        "interior_footprint_sq_ft": round(float(footprint_m2) * _SQ_FT_PER_SQ_M, 1),
    }


def _read_cloud(zip_path, model_id: str) -> tuple[np.ndarray, str | None]:
    """Parse cloud.xyz (x y z r g b per line) from the bundle and keep its floor-plan image."""
    with zipfile.ZipFile(zip_path) as bundle:
        names = bundle.namelist()
        member = (next((n for n in names if n.lower().endswith("cloud.xyz")), None)
                  or next((n for n in names if n.lower().endswith(".xyz")), None))
        if member is None:
            raise ValueError("bundle has no .xyz point cloud")
        with bundle.open(member) as handle:
            frame = pd.read_csv(handle, sep=" ", header=None, usecols=[0, 1, 2], dtype=np.float32,
                                engine="c", na_filter=False)
        plan = next((n for n in names if n.lower().startswith("colorplan") and n.lower().endswith(".jpg")), None)
        plan_path = None
        if plan:
            target = _floorplan_path(model_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(plan))
            plan_path = str(target)
    return frame.to_numpy(), plan_path


def get_matterpak(model_id: str, download: bool = False) -> MatterpakGeometry:
    """Interior height/footprint from the model's MatterPak point cloud, cached per model.

    The zip is fetched only when ``download`` is true (the showcase batch does this
    for ranked candidates). A study run passes False and gets ``available=False``
    until a bundle has been fetched, so no request ever waits on a 75 MB download.
    """
    key = cache_key(NAMESPACE, {"model_id": model_id, "geometry": _GEOMETRY_VERSION})
    try:
        cached = load_cached(key, NAMESPACE)
    except CacheMiss:
        cached = None
    if cached is not None:
        return MatterpakGeometry.model_validate(cached)
    path = _bundle_path(model_id)
    if not path.exists():
        if not download or config.USE_CACHE_ONLY:
            return MatterpakGeometry(model_id=model_id, reason="MatterPak bundle not downloaded yet")
        try:
            url = _bundle_url(model_id)
            if not url:
                result = MatterpakGeometry(model_id=model_id, reason="no MatterPak bundle is available for this model")
                save_cached(key, result.model_dump(mode="json"), NAMESPACE)
                return result
            _download_bundle(url, path, MATTERPAK_MAX_MB * 1024 * 1024)
        except ValueError as exc:  # the size guard
            result = MatterpakGeometry(model_id=model_id, reason=f"skipped: {exc}")
            save_cached(key, result.model_dump(mode="json"), NAMESPACE)
            return result
    points, plan_path = _read_cloud(path, model_id)
    result = MatterpakGeometry(model_id=model_id, available=True, bundle_bytes=path.stat().st_size,
                               floorplan_image_path=plan_path, **geometry_from_xyz(points))
    save_cached(key, result.model_dump(mode="json"), NAMESPACE)
    return result
