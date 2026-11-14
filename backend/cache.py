"""Cache interfaces for data/cache; persistence is not implemented yet."""

from pydantic import JsonValue


def load_cached(key: str) -> JsonValue | None:
    """TODO: Load a JSON value from CACHE_DIR, or return None on a miss."""
    ...


def save_cached(key: str, value: JsonValue) -> None:
    """TODO: Persist a JSON value in CACHE_DIR without storing credentials."""
    ...
