"""Verify Vexcel request coordinates, caching, retry behavior and field coverage."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import requests
from shapely import wkt

from scripts.vexcel_test import (
    BASE_URL,
    CachedVexcelClient,
    field_paths,
    field_value,
    request_payloads,
    run_probe,
    select_properties,
    write_field_report,
)


def response(status: int, content: bytes, content_type: str = "application/json") -> requests.Response:
    """Build a local HTTP response without contacting Vexcel."""
    result = requests.Response()
    result.status_code = status
    result._content = content
    result.headers["Content-Type"] = content_type
    return result


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    """Create eligible, wrong-market, wrong-type and ungeocoded properties."""
    records = [
        {"id": i, "propertyType": 11, "primaryAddress": {
            "researchMarketId": 14, "latitude": 29.75 + i * 0.001, "longitude": -95.4,
        }} for i in (3, 1, 2)
    ]
    records.extend([
        {"id": 4, "propertyType": 5, "primaryAddress": {"researchMarketId": 14, "latitude": 29, "longitude": -95}},
        {"id": 5, "propertyType": 11, "primaryAddress": {"researchMarketId": 20, "latitude": 34, "longitude": -118}},
        {"id": 6, "propertyType": 11, "primaryAddress": {"researchMarketId": 14, "latitude": None, "longitude": -95}},
    ])
    path = tmp_path / "commercial-property.jsonl"
    pd.DataFrame(records).to_json(path, orient="records", lines=True)
    return path


def test_selection_and_longitude_first_square(kit: Path) -> None:
    """Choose three distinct Houston apartments and preserve an 80 m square AOI."""
    assert [record["id"] for record in select_properties(kit)] == [1, 2, 3]
    params, body = request_payloads(29.75, -95.4)
    assert params["wkt"] == "POINT(-95.4 29.75)"
    assert params["srid"] == 4326 and params["latest"] == "true"
    assert params["include"] == "precomputed" and params["unit-system"] == "imperial"
    square = wkt.loads(body["wkt"])
    west, south, east, north = square.bounds
    assert (east - west) * 111_320 * math.cos(math.radians(29.75)) == pytest.approx(80)
    assert (north - south) * 111_320 == pytest.approx(80)
    assert square.centroid.x == pytest.approx(-95.4)
    assert square.centroid.y == pytest.approx(29.75)
    assert body["buffer-meter"] == 10 and body["crop"] == "bounding_box"
    assert body["image-format"] == "jpeg" and body["drawing"][0]["wkt"] == body["wkt"]


def test_retry_headers_and_secret_free_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Retry 429s, send credentials only in headers, and reuse the successful response."""
    calls = []
    sleeps = []
    outcomes = [response(429, b'{"error":"rate limited"}'), response(200, b'{"features":[]}')]

    def send(method: str, url: str, **kwargs: object) -> requests.Response:
        """Capture an HTTP attempt and return the next synthetic response."""
        calls.append((method, url, kwargs))
        return outcomes.pop(0)

    client = CachedVexcelClient("private-test-sentinel", tmp_path, send=send, sleep=sleeps.append, jitter=lambda: 0)
    result = client.request("GET", "property/extract", {"srid": 4326}, 120)
    assert result.status_code == 200
    assert sleeps == [1] and len(calls) == 2
    assert calls[0][1] == BASE_URL + "/property/extract"
    assert calls[0][2]["headers"] == {"X-API-KEY": "private-test-sentinel"}
    assert calls[0][2]["params"] == {"srid": 4326}
    assert calls[0][2]["allow_redirects"] is False
    client.request("GET", "property/extract", {"srid": 4326}, 120)
    assert len(calls) == 2
    assert capsys.readouterr().out.splitlines() == ["429", "200", "200"]
    assert len(list(tmp_path.glob("*-retry-*.json"))) == 1
    assert all("private-test-sentinel" not in path.read_text() for path in tmp_path.glob("*.json"))


def test_cache_only_never_calls_network(tmp_path: Path) -> None:
    """An offline miss is explicit and cannot spend a live API request."""
    def forbidden(*args: object, **kwargs: object) -> requests.Response:
        """Fail if cache-only code attempts a network call."""
        pytest.fail("Network called in cache-only mode")

    client = CachedVexcelClient(None, tmp_path, cache_only=True, send=forbidden)
    with pytest.raises(RuntimeError, match="cache_miss"):
        client.request("GET", "property/extract", {}, 120)


def test_failed_auth_cache_does_not_poison_later_live_requests(tmp_path: Path) -> None:
    """After correcting authentication, a new call can replace a cached 401."""
    outcomes = [response(401, b'{}'), response(200, b'{"features":[]}')]
    client = CachedVexcelClient("test-sentinel", tmp_path, send=lambda *a, **kw: outcomes.pop(0))
    assert client.request("GET", "property/extract", {}, 120).status_code == 401
    assert client.request("GET", "property/extract", {}, 120).status_code == 200


def test_all_leaf_fields_and_array_values_preserved(tmp_path: Path) -> None:
    """Inventory dotted keys, multiple structures, geometry arrays, null and false."""
    data = {"features": [{"properties": {
        "pool.detected": False,
        "parcel-tree-cover.percent": 0,
        "nullable": None,
        "structures": [{"footprint.area": 10}, {"footprint.area": 20}],
    }, "geometry": {"coordinates": [[[1, 2], [3, 4]]]}}]}
    path = ("features", None, "properties", "structures", None, "footprint.area")
    assert path in field_paths(data)
    assert field_value(data, path) == [[10, 20]]
    result = {"property": {"id": 1, "latitude": 29, "longitude": -95},
              "property_status": 200, "image_status": 200, "response": data}
    output = tmp_path / "vexcel_fields.md"
    write_field_report([result, {"property": {"id": 2, "latitude": 29, "longitude": -95}}], output)
    text = output.read_text()
    assert '[[10, 20]]' in text and '[false]' in text and '[null]' in text
    assert '[[[[1, 2], [3, 4]]]]' in text
    assert "<not returned>" in text and '["footprint.area"]' in text
    for leaf in field_paths(data):
        assert field_value(data, leaf) is not None or leaf == ()


def test_probe_saves_six_artifacts_and_status_only_output(
    kit: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """GET JSON and POST images use the requested paths and payload types."""
    calls = []

    def send(method: str, url: str, **kwargs: object) -> requests.Response:
        """Return a sample precomputed response or JPEG bytes for the probe."""
        calls.append((method, url, kwargs))
        if method == "GET":
            return response(200, json.dumps({"features": [{"properties": {"pool.detected": False}}]}).encode())
        return response(200, b"\xff\xd8\xffexample", "image/jpeg")

    samples = tmp_path / "samples"
    report = tmp_path / "vexcel_fields.md"
    client = CachedVexcelClient("private-test-sentinel", tmp_path / "cache", send=send)
    assert run_probe(client, kit, samples, report)
    assert len(list(samples.glob("vexcel_property_*.json"))) == 3
    assert len(list(samples.glob("vexcel_*.jpg"))) == 3
    assert len(calls) == 6 and all("json" in kwargs for method, _, kwargs in calls if method == "POST")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 13
    assert all(line == "200" or line.endswith((".json", ".jpg", ".md")) for line in lines)
    assert "private-test-sentinel" not in report.read_text()


def test_probe_continues_imagery_after_404_and_records_unavailability(kit: Path, tmp_path: Path) -> None:
    """No property data is not a reported zero and does not prevent square imagery."""
    def send(method: str, url: str, **kwargs: object) -> requests.Response:
        """Return no parcel JSON, but a valid JPEG signature for each square."""
        return response(404, b'{}') if method == "GET" else response(200, b"\xff\xd8\xffexample", "image/jpeg")

    client = CachedVexcelClient("test-sentinel", tmp_path / "cache", send=send)
    output = tmp_path / "vexcel_fields.md"
    assert not run_probe(client, kit, tmp_path / "samples", output)
    assert len(list((tmp_path / "samples").glob("*.jpg"))) == 3
    assert "No successful precomputed response" in output.read_text()


def test_non_jpeg_body_cannot_be_saved_as_image(kit: Path, tmp_path: Path) -> None:
    """An HTTP 200 JSON body from the image endpoint must not become a .jpg."""
    client = CachedVexcelClient("test-sentinel", tmp_path / "cache", send=lambda *a, **kw: response(200, b'{"features":[]}'))
    assert not run_probe(client, kit, tmp_path / "samples", tmp_path / "report.md")
    assert not list((tmp_path / "samples").glob("*.jpg"))
