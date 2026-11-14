"""Vexcel API stubs: X-API-KEY auth, longitude-first WKT, SRID 4326."""

from backend.schemas import AerialImage, VexcelPropertyInfo


def get_aerial(lat: float, lon: float, property_id: int) -> AerialImage:
    """TODO: Cache every Vexcel call and return JSON metadata for saved imagery."""
    # TODO: Honor USE_CACHE_ONLY and use VEXCEL_API_KEY with VEXCEL_BASE_URL.
    raise NotImplementedError


def get_property_info(lat: float, lon: float) -> VexcelPropertyInfo:
    """TODO: Cache every property/extract call, honoring USE_CACHE_ONLY."""
    # TODO: Send longitude-first WKT and srid=4326 with X-API-KEY auth.
    raise NotImplementedError
