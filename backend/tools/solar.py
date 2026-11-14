"""Rooftop solar estimation and ranking interfaces."""

from backend.schemas import Building, Measurement, SolarResult


def estimate_solar(
    building: Building, vision_result: Measurement
) -> SolarResult:
    """TODO: Estimate rooftop solar potential with explicit units and assumptions."""
    raise NotImplementedError


def rank_buildings(
    estimates: list[SolarResult],
) -> list[SolarResult]:
    """TODO: Rank buildings by a documented rooftop solar metric."""
    raise NotImplementedError
