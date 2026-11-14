"""Vexcel API stubs: X-API-KEY auth, longitude-first WKT, SRID 4326."""

from pydantic import JsonValue


def get_aerial(lat: float, lon: float, property_id: int) -> dict[str, JsonValue]:
    """TODO: Cache every Vexcel call and return JSON metadata for saved imagery."""
    # TODO: Honor USE_CACHE_ONLY and use VEXCEL_API_KEY with VEXCEL_BASE_URL.
    ...


def get_property_info(lat: float, lon: float) -> dict[str, JsonValue]:
    """TODO: Cache every property/extract call, honoring USE_CACHE_ONLY."""
    # TODO: Send longitude-first WKT and srid=4326 with X-API-KEY auth.
    ...
