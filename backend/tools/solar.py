"""Rooftop solar estimation and ranking interfaces."""

from pydantic import JsonValue


def estimate_solar(
    building: dict[str, JsonValue], vision_result: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """TODO: Estimate rooftop solar potential with explicit units and assumptions."""
    ...


def rank_buildings(
    estimates: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """TODO: Rank buildings by a documented rooftop solar metric."""
    ...
