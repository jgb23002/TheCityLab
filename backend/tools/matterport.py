"""Optional Matterport GraphQL integration using HTTP Basic authentication.

Room-level dimensionEstimates are a real measurement (height, area, lower
elevation), not a vision model's photo reading, so a building's height may
prefer this over the CoStar-math fallback in backend.tools.geometry when a
matterport_model_id is known and its rooms report a usable height.
"""

from __future__ import annotations

import statistics
from typing import Any

import requests

from backend import config
from backend.cache import cached_json_call
from backend.schemas import MatterportModel

_QUERY = """
query getRoomDims($modelId: ID!) {
  model(id: $modelId) {
    rooms {
      id
      label
      floor { id }
      dimensionEstimates {
        area
        areaIndoor
        width
        depth
        height
        lowerElevation
        units
      }
    }
    floors {
      id
      label
      sequence
      dimensionEstimates { area areaIndoor units }
    }
  }
}
"""


def _fetch(model_id: str) -> dict[str, Any]:
    """Send the GraphQL query for one model id; never logs credentials."""
    if not config.MP_TOKEN_ID or not config.MP_TOKEN_SECRET:
        raise ValueError("Set MP_TOKEN_ID and MP_TOKEN_SECRET in .env for Matterport.")
    response = requests.post(
        config.MATTERPORT_GRAPHQL_URL,
        json={"query": _QUERY, "variables": {"modelId": model_id}},
        auth=(config.MP_TOKEN_ID, config.MP_TOKEN_SECRET),
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()
    if "errors" in body:
        raise ValueError(f"Matterport GraphQL errors for model {model_id!r}: {body['errors']}")
    return body


def get_model(model_id: str) -> MatterportModel:
    """Fetch one model's room/floor dimension estimates, cached by model id."""
    body = cached_json_call("matterport", {"model_id": model_id}, lambda: _fetch(model_id))
    model = (body.get("data") or {}).get("model") or {}
    return MatterportModel(id=model_id, metadata={"rooms": model.get("rooms"), "floors": model.get("floors")},
                           raw_response=body)


def building_height_ft(model: MatterportModel) -> float | None:
    """Return the median of rooms' positive dimensionEstimates.height, or None.

    Matterport's room scan sometimes can't resolve a ceiling plane (height
    comes back 0.0) for large or unusually shaped rooms; those are excluded
    rather than treated as a real zero-height room.
    """
    rooms = (model.metadata or {}).get("rooms") or []
    if not isinstance(rooms, list):
        return None
    heights = [
        height for room in rooms
        if isinstance(room, dict)
        and isinstance((dims := room.get("dimensionEstimates")), dict)
        and isinstance((height := dims.get("height")), (int, float))
        and height > 0.5
    ]
    return statistics.median(heights) if heights else None
