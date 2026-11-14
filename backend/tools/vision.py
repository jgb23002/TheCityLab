"""Aerial building measurement interfaces; no vision calls are implemented."""

from backend.schemas import Measurement


def measure_building(image_path: str, property_id: int) -> Measurement:
    """TODO: Measure the building from a cached aerial image with confidence."""
    raise NotImplementedError
