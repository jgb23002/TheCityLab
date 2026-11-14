"""Disk-backed JSON and binary cache under data/cache/.

Every Vexcel and CoStar-adjacent call is routed through this module so that
demo mode (USE_CACHE_ONLY=true) never touches the network.

Usage::

    from backend.cache import cached_json_call, load_cached, save_cached

    result = cached_json_call("vexcel/property", {"lat": 29.68, "lon": -95.46},
                              lambda: live_api_call(...))
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from backend import config

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class CacheMiss(RuntimeError):
    """Raised when USE_CACHE_ONLY=true and there is no cached response."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(namespace: str, key: str, suffix: str) -> Path:
    """Return the cache file path, creating parent directories as needed."""
    path = config.CACHE_DIR / namespace / f"{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def cache_key(namespace: str, params: dict[str, Any]) -> str:
    """Return a stable 16-character hex digest for *namespace* + *params*.

    The digest is derived from the SHA-256 of the JSON-serialised params with
    keys sorted, so the same logical request always maps to the same file.
    """
    raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# JSON cache
# ---------------------------------------------------------------------------

def load_cached(key: str, namespace: str = "default") -> JsonValue | None:
    """Return the cached JSON value for *key* inside *namespace*, or None.

    If USE_CACHE_ONLY is true and the cache file does not exist, raises
    CacheMiss instead of returning None.
    """
    path = _resolve(namespace, key, ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if config.USE_CACHE_ONLY:
        raise CacheMiss(f"Cache miss in '{namespace}' for key '{key}' and USE_CACHE_ONLY=true")
    return None


def save_cached(key: str, value: JsonValue, namespace: str = "default") -> None:
    """Persist *value* as JSON under *namespace/key*.json.

    Credentials must never be part of *value*; callers are responsible for
    stripping them before calling this function.
    """
    path = _resolve(namespace, key, ".json")
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def cached_json_call(
    namespace: str,
    params: dict[str, Any],
    fn: Callable[[], JsonValue],
) -> JsonValue:
    """Return the cached result for *params* in *namespace*, or call *fn*.

    On a cache miss the result of *fn()* is saved before it is returned.
    If USE_CACHE_ONLY is true, *fn* is never invoked; a CacheMiss is raised.
    """
    key = cache_key(namespace, params)
    cached = load_cached(key, namespace)
    if cached is not None:
        return cached
    result = fn()
    save_cached(key, result, namespace)
    return result


# ---------------------------------------------------------------------------
# Binary (image) cache
# ---------------------------------------------------------------------------

def load_bytes(key: str, namespace: str = "images") -> bytes | None:
    """Return raw bytes from *namespace/key*.bin, or None on a miss."""
    path = _resolve(namespace, key, ".bin")
    if path.exists():
        return path.read_bytes()
    if config.USE_CACHE_ONLY:
        raise CacheMiss(f"Cache miss in '{namespace}' for key '{key}' and USE_CACHE_ONLY=true")
    return None


def save_bytes(key: str, data: bytes, namespace: str = "images") -> Path:
    """Write *data* to *namespace/key*.bin and return the file path."""
    path = _resolve(namespace, key, ".bin")
    path.write_bytes(data)
    return path
