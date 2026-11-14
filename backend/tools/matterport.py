"""Optional Matterport GraphQL integration using HTTP Basic authentication.

``get_model`` returns the scan's measured spatial metadata (floors and rooms with
their measured dimensions). The Matterport Model API used here exposes no interior
images, so lighting, window and HVAC observations are never inferred from it.
Every response is cached under data/cache/matterport/ and USE_CACHE_ONLY is honoured.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from typing import Any

import requests

from backend import config
from backend.cache import CacheMiss, cache_key, load_cached, save_cached
from backend.schemas import MatterportModel

NAMESPACE = "matterport"
_MODEL_QUERY = """query($id: ID!) {
  model(id: $id) {
    id name created visibility
    floors { id label dimensions { areaFloor units } }
    rooms { id label tags dimensions { height areaFloor width depth units } }
  }
}"""


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


def summarize(model: dict[str, Any]) -> dict[str, Any]:
    """Measured facts only: floor and room counts, floor areas, measured ceiling heights."""
    floors = [f for f in model.get("floors") or [] if isinstance(f, dict)]
    rooms = [r for r in model.get("rooms") or [] if isinstance(r, dict)]
    floor_area = [((f.get("dimensions") or {}).get("areaFloor")) for f in floors]
    heights = [((r.get("dimensions") or {}).get("height")) for r in rooms]
    heights = [h for h in heights if isinstance(h, (int, float)) and h > 0]
    units = next(((f.get("dimensions") or {}).get("units") for f in floors if f.get("dimensions")), None)
    return {
        "name": model.get("name"),
        "created": model.get("created"),
        "floor_count": len(floors),
        "scanned_floor_area": round(sum(a for a in floor_area if isinstance(a, (int, float))), 1) if floor_area else None,
        "room_count": len(rooms),
        "labelled_rooms": sorted({r.get("label") for r in rooms if r.get("label")}),
        "measured_ceiling_heights": [round(h, 2) for h in heights],
        "units": units,
        "rooms": [{"label": r.get("label") or None, "tags": r.get("tags") or [],
                   "area_floor": (r.get("dimensions") or {}).get("areaFloor"),
                   "width": (r.get("dimensions") or {}).get("width"),
                   "depth": (r.get("dimensions") or {}).get("depth"),
                   "height": (r.get("dimensions") or {}).get("height")} for r in rooms],
    }


def get_model(model_id: str) -> MatterportModel:
    """Fetch (or replay from cache) one Matterport model's measured spatial metadata."""
    key = cache_key(NAMESPACE, {"model_id": model_id, "query": "spatial-v1"})
    cached = load_cached(key, NAMESPACE)  # raises CacheMiss when USE_CACHE_ONLY and not cached
    if cached is None:
        if config.USE_CACHE_ONLY:  # pragma: no cover - load_cached raises first
            raise CacheMiss(f"Matterport cache miss for {model_id}")
        cached = _fetch(model_id)
        save_cached(key, cached, NAMESPACE)
    return MatterportModel(id=model_id, metadata=summarize(cached), raw_response=None)
