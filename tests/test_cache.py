"""Tests for backend/cache.py."""

import json
from pathlib import Path

import pytest

import backend.config as cfg
from backend.cache import (
    CacheMiss,
    cache_key,
    cached_json_call,
    load_bytes,
    load_cached,
    save_bytes,
    save_cached,
)


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_stable() -> None:
    """The same namespace + params always produce the same key."""
    k1 = cache_key("vexcel", {"lat": 29.7, "lon": -95.4})
    k2 = cache_key("vexcel", {"lon": -95.4, "lat": 29.7})  # different order
    assert k1 == k2


def test_cache_key_different_namespaces() -> None:
    """Different namespaces produce different keys for the same params."""
    k1 = cache_key("ns_a", {"x": 1})
    k2 = cache_key("ns_b", {"x": 1})
    assert k1 != k2


def test_cache_key_length() -> None:
    """cache_key returns a 16-character hex string."""
    k = cache_key("test", {"a": "b"})
    assert len(k) == 16
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# JSON cache
# ---------------------------------------------------------------------------


def test_save_and_load_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_cached / load_cached round-trips a JSON value."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    save_cached("mykey", {"answer": 42}, namespace="test_ns")
    result = load_cached("mykey", namespace="test_ns")
    assert result == {"answer": 42}


def test_load_missing_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_cached returns None when no file exists and USE_CACHE_ONLY is false."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", False)
    assert load_cached("no_such_key", namespace="test_ns") is None


def test_load_missing_cache_only_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_cached raises CacheMiss when USE_CACHE_ONLY=true and key is absent."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", True)
    with pytest.raises(CacheMiss):
        load_cached("absent_key", namespace="test_ns")


def test_cached_json_call_calls_fn_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cached_json_call invokes fn() on a miss and caches the result."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", False)
    calls: list[int] = []

    def fn() -> dict:
        calls.append(1)
        return {"v": 7}

    result = cached_json_call("my_ns", {"x": 1}, fn)
    assert result == {"v": 7}
    assert calls == [1]

    # Second call should NOT invoke fn
    result2 = cached_json_call("my_ns", {"x": 1}, fn)
    assert result2 == {"v": 7}
    assert calls == [1]  # still only one call


def test_cached_json_call_cache_only_miss_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cached_json_call raises CacheMiss when cache_only=true and nothing cached."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", True)

    with pytest.raises(CacheMiss):
        cached_json_call("my_ns", {"y": 2}, lambda: {"should": "not reach"})


# ---------------------------------------------------------------------------
# Binary cache
# ---------------------------------------------------------------------------


def test_save_and_load_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_bytes / load_bytes round-trips binary data."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    data = b"\xff\xd8\xff\xe0JPEG"
    save_bytes("imgkey", data, namespace="imgs")
    assert load_bytes("imgkey", namespace="imgs") == data


def test_load_bytes_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_bytes returns None when no file exists and USE_CACHE_ONLY is false."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", False)
    assert load_bytes("no_img", namespace="imgs") is None


def test_load_bytes_cache_only_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_bytes raises CacheMiss when USE_CACHE_ONLY=true and key is absent."""
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "USE_CACHE_ONLY", True)
    with pytest.raises(CacheMiss):
        load_bytes("absent_img", namespace="imgs")

