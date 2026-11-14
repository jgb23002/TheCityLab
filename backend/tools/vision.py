"""Aerial building measurement interfaces; no vision calls are implemented."""

from pydantic import JsonValue


def measure_building(image_path: str, property_id: int) -> dict[str, JsonValue]:
    """TODO: Measure the building from a cached aerial image with confidence."""
    ...
