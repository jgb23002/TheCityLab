"""Roof footprint, building height and lot geometry from CoStar fields alone.

Pure arithmetic, no network and no model calls: per the project's "math first,
model last" rule, anything a formula over CoStar's own fields can answer is
computed here instead of asked of a vision model. Every function returns
``None`` (never 0 or a guess) when its required source fields are missing.
"""

from __future__ import annotations

from backend.schemas import Building

# CoStar's ceilingHeightFeet is null on every sampled multifamily record, so a
# per-floor height has to be a stated assumption rather than a measurement.
FLOOR_TO_FLOOR_FT = 12.0

# lotSize is already unit-converted by CoStar except when flagged Acres (2);
# SquareFeet (1) and Unknown (0) are left as-is. See docs/CONTRACTS.md.
_ACRES_TYPE = 2
SQ_FT_PER_ACRE = 43_560.0


def roof_footprint_sq_ft(building: Building) -> float | None:
    """Return the building's roof footprint, falling back through 3 CoStar fields.

    1. ``gross_typical_floor_sq_ft`` (CoStar's own "most common floor" metric).
    2. The largest value in ``floor_areas_sq_ft`` (per-floor areas).
    3. ``building_size_sq_ft`` / ``num_stories_above_grade`` (GBA divided by stories).
    """
    if building.gross_typical_floor_sq_ft and building.gross_typical_floor_sq_ft > 0:
        return building.gross_typical_floor_sq_ft
    if building.floor_areas_sq_ft:
        return max(building.floor_areas_sq_ft)
    if (
        building.building_size_sq_ft and building.building_size_sq_ft > 0
        and building.num_stories_above_grade and building.num_stories_above_grade > 0
    ):
        return building.building_size_sq_ft / building.num_stories_above_grade
    return None


def building_height_ft(building: Building) -> float | None:
    """Return building height: CoStar's own measurement when present, else stories * assumed floor height."""
    if building.building_height_ft and building.building_height_ft > 0:
        return building.building_height_ft
    if building.num_stories_above_grade and building.num_stories_above_grade > 0:
        return building.num_stories_above_grade * FLOOR_TO_FLOOR_FT
    return None


def lot_size_sq_ft(building: Building) -> float | None:
    """Return lot size in square feet, converting from acres when flagged as such."""
    if not building.lot_size or building.lot_size <= 0:
        return None
    if building.lot_size_measurement_type == _ACRES_TYPE:
        return building.lot_size * SQ_FT_PER_ACRE
    return building.lot_size


def open_ground_sq_ft(building: Building) -> float | None:
    """Return lot area minus roof footprint, or None if either input is unknown."""
    lot = lot_size_sq_ft(building)
    footprint = roof_footprint_sq_ft(building)
    if lot is None or footprint is None:
        return None
    return lot - footprint
