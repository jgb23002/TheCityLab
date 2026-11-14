"""Fetch three Houston apartment examples and inventory Vexcel response fields.

Run ``python scripts/vexcel_test.py``. Credentials come from .env. All calls are
cached locally; USE_CACHE_ONLY forbids network requests. Console output contains
only HTTP status codes and saved filenames. Local tutorial files are not copied.
"""

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from dotenv import load_dotenv
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "costar" / "data" / "commercial-property.jsonl"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "vexcel"
REPORT_PATH = PROJECT_ROOT / "notes" / "vexcel_fields.md"
BASE_URL = "https://api.vexcelgroup.com/v2"
LAYERS = "urban,urban-r,wide-area,urban-g"
MAX_RETRIES = 5
FieldPath = tuple[str | None, ...]
ABSENT = "<not returned>"


class CachedVexcelClient:
    """Use header authentication, cache responses and retry 429s with backoff."""

    def __init__(
        self, api_key: str | None, cache_dir: Path = CACHE_DIR, cache_only: bool = False,
        send: Callable[..., requests.Response] = requests.request,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Keep credentials in memory; the cache identity never includes headers."""
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_only = cache_only
        self.send = send
        self.sleep = sleep
        self.jitter = jitter

    def request(
        self, method: str, endpoint: str, payload: dict[str, object], timeout: int,
    ) -> requests.Response:
        """Return a cached response or cache each HTTP attempt without logging bodies."""
        identity = {"method": method, "endpoint": endpoint, "payload": payload}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        metadata_path = self.cache_dir / f"{digest}.json"
        body_path = self.cache_dir / f"{digest}.body"
        if metadata_path.exists() and body_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if self.cache_only or 200 <= metadata["status_code"] < 300:
                response = requests.Response()
                response.status_code = metadata["status_code"]
                response._content = body_path.read_bytes()
                response.headers["Content-Type"] = metadata.get("content_type", "")
                print(response.status_code, flush=True)
                return response
        if self.cache_only:
            raise RuntimeError("cache_miss")
        if not self.api_key:
            raise RuntimeError("missing_credential")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(MAX_RETRIES + 1):
            kwargs = {"params" if method == "GET" else "json": payload}
            response = self.send(
                method, f"{BASE_URL}/{endpoint}", headers={"X-API-KEY": self.api_key},
                timeout=timeout, allow_redirects=False, **kwargs,
            )
            print(response.status_code, flush=True)
            metadata = {**identity, "status_code": response.status_code,
                        "content_type": response.headers.get("Content-Type", "")}
            body_path.write_bytes(response.content)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            # Preserve rate-limit attempts too, while the main cache tracks the final response.
            if response.status_code == 429:
                retry_path = self.cache_dir / f"{digest}-retry-{time.time_ns()}"
                retry_path.with_suffix(".body").write_bytes(response.content)
                retry_path.with_suffix(".json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            if response.status_code != 429 or attempt == MAX_RETRIES:
                return response
            self.sleep(min(2 ** attempt, 30) + self.jitter())
        raise RuntimeError("retry_limit")


def select_properties(data_path: Path = DATA_PATH, count: int = 3) -> list[dict[str, object]]:
    """Select the lowest distinct ids with Houston/Multifamily codes and coordinates."""
    frame = pd.read_json(data_path, lines=True)
    selected = {}
    for row in frame.loc[frame["propertyType"].eq(11)].to_dict(orient="records"):
        address = row.get("primaryAddress")
        if not isinstance(address, dict) or address.get("researchMarketId") != 14:
            continue
        try:
            lat, lon = float(address.get("latitude")), float(address.get("longitude"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(lat) and math.isfinite(lon) and -90 < lat < 90 and -180 <= lon <= 180:
            property_id = int(row["id"])
            selected[property_id] = {"id": property_id, "latitude": lat, "longitude": lon}
    if len(selected) < count:
        raise ValueError("Not enough geocoded Houston apartment properties.")
    return [selected[property_id] for property_id in sorted(selected)[:count]]


def request_payloads(lat: float, lon: float) -> tuple[dict[str, object], dict[str, object]]:
    """Build longitude-first point and approximately 80 m square requests at SRID 4326."""
    delta_lat = 40 / 111_320
    delta_lon = 40 / (111_320 * math.cos(math.radians(lat)))
    square = box(lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat).wkt
    property_params = {
        "wkt": f"POINT({lon} {lat})", "srid": 4326, "layer": LAYERS,
        "latest": "true", "include": "precomputed", "unit-system": "imperial",
    }
    image_body = {
        "wkt": square, "srid": 4326, "layer": LAYERS, "buffer-meter": 10,
        "crop": "bounding_box", "image-format": "jpeg",
        "drawing": [{"wkt": square, "outline-width": 4, "outline-color": "rgb(255,204,0)"}],
    }
    return property_params, image_body


def field_paths(value: object, prefix: FieldPath = ()) -> set[FieldPath]:
    """Collect all leaf paths, preserving dotted keys and arrays of geometrical values."""
    if isinstance(value, dict) and value:
        return set().union(*(field_paths(child, (*prefix, key)) for key, child in value.items()))
    if isinstance(value, list) and value and any(isinstance(child, dict) for child in value):
        return set().union(*(field_paths(child, (*prefix, None)) for child in value))
    return {prefix}


def field_value(value: object, path: FieldPath) -> object:
    """Keep array order and distinguish explicit JSON null from an absent field."""
    if not path:
        return value
    token, *remaining = path
    if token is None:
        return [field_value(item, tuple(remaining)) for item in value] if isinstance(value, list) else ABSENT
    if not isinstance(value, dict) or token not in value:
        return ABSENT
    return field_value(value[token], tuple(remaining))


def path_label(path: FieldPath) -> str:
    """Render an exact JSON-style path; dotted attribute names remain literal keys."""
    return "$" + "".join("[*]" if part is None else f"[{json.dumps(part)}]" for part in path)


def field_meaning(path: FieldPath) -> str:
    """Use tutorial definitions or clearly labeled interpretations of provider keys."""
    name = next((part for part in reversed(path) if part is not None), "response")
    documented = {
        "type": "GeoJSON object or geometry type.",
        "id": "Provider identifier for this feature or object.",
        "coordinates": "Geometry coordinates; geographic positions use longitude then latitude (SRID 4326).",
        "metadata.collection": "Vexcel imagery collection identifier.",
        "structure.count": "Number of structures/buildings returned for this parcel.",
        "structures": "Structures array; an empty array means no structures were returned.",
        "primary-structure": "Provider flag identifying the parcel's primary structure.",
        "parcel-tree-cover.percent": "Percentage of the Vexcel parcel covered by trees; not all vegetation/green cover.",
        "pool.detected": "Whether the precomputed model detected a pool on the parcel.",
        "pool.total-area": "Total detected pool area; square feet for this imperial request.",
        "footprint.area": "Structure footprint area in square feet; not total floor area or roof surface area.",
        "building-type.class": "Provider's inferred building-use class, which need not match CoStar propertyType.",
        "roof-characteristics.material.class": "Predicted roof covering material class.",
        "roof-characteristics.shape.value": "Predicted roof shape category.",
        "roof-characteristics.condition": "Provider roof-condition score; the local tutorial does not define the full scale or direction.",
        "request-information.inside-parcel": "Whether the submitted point falls inside the returned parcel.",
    }
    if name in documented:
        return documented[name]
    human = name.replace("-", " ").replace(".", " / ").replace("_", " ")
    if name.endswith(".percent"):
        return f"{human}: a percentage; the precise denominator is provider-defined (interpretation from key)."
    if name.endswith(".count"):
        return f"{human}: a detected object count (interpretation from key); vehicle counts are not parking-space counts."
    if name.endswith(".detected"):
        return f"Whether {human.removesuffix(' / detected')} was detected (interpretation from key)."
    if name.endswith((".area", ".total-area")):
        return f"{human}: area, requested in imperial units; confirm any response-specific unit (interpretation from key)."
    if name.endswith((".wkt", "-wkt")):
        return f"{human}: geometry encoded as Well-Known Text (interpretation from key)."
    if name.endswith(".confidence"):
        return f"Provider confidence for {human.removesuffix(' / confidence')}; scale not defined by the local tutorial."
    return f"Provider field for {human} (interpretation from key); exact unit/category scale is not defined in the local tutorial."


def markdown_value(value: object) -> str:
    """Preserve complete JSON values in a table cell without truncating arrays."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("|", "&#124;").replace("`", "&#96;")


def write_field_report(results: list[dict[str, object]], output_path: Path = REPORT_PATH) -> None:
    """Inventory every returned leaf field, including metadata and full geometry values."""
    lines = [
        "# Vexcel precomputed fields", "",
        "Generated by `python scripts/vexcel_test.py` for three Houston CoStar apartment properties "
        "(`propertyType == 11`, `primaryAddress.researchMarketId == 14`), selected by ascending property id with coordinates.",
        "The local quickstart and 13-page tutorial were read for request conventions and field meanings; "
        "neither reference file is copied into the repository.",
        "", "## Requests and saved artifacts", "",
        "GET `property/extract`: longitude-first `POINT(lon lat)`, SRID 4326, latest=true, "
        "include=precomputed, unit-system=imperial, layers urban/urban-r/wide-area/urban-g.",
        "POST `ortho/extract`: an approximately 80 m square constructed with shapely.box using the quickstart's "
        "local metres-to-degrees approximation, plus its 10 m context buffer, bounding_box crop, JPEG, and gold outline.",
        "Responses are cached under git-ignored `data/cache/vexcel/`. USE_CACHE_ONLY prohibits network calls. "
        "Requests use only the X-API-KEY header for authentication, with up to five exponential-backoff retries on 429.",
        "", "| CoStar property id | Latitude | Longitude | Property HTTP status | Image HTTP status | JSON | Image |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for result in results:
        prop = result["property"]
        lines.append(
            f"| {prop['id']} | {prop['latitude']} | {prop['longitude']} | "
            f"{result.get('property_status') or result.get('property_error', 'not requested')} | "
            f"{result.get('image_status') or result.get('image_error', 'not requested')} | "
            f"{result.get('json_file', 'not saved')} | {result.get('image_file', 'not saved')} |"
        )
    json_count = sum(result.get("response") is not None for result in results)
    image_count = sum("image_file" in result for result in results)
    lines.extend([
        "", f"Successful precomputed responses: {json_count}/{len(results)}; "
        f"saved JPEGs: {image_count}/{len(results)}.",
    ])
    if any(result.get("property_status") == 401 or result.get("image_status") == 401 for result in results):
        lines.extend([
            "", "HTTP 401 means Vexcel rejected authentication. Renew or replace VEXCEL_API_KEY in the local .env "
            "and rerun this script; failed responses are cached for diagnosis but do not prevent a new live attempt. "
            "No precomputed values or successful image files are invented for failed requests.",
        ])
    lines.extend([
        "", "## Complete field inventory", "",
        "Paths use bracket notation so dots inside Vexcel attribute names are literal. "
        "`[*]` preserves feature/structure order as nested arrays in each value. "
        "All leaf fields (including empty containers), metadata and geometry values are included without truncation.",
        "JSON null means reported null; `<not returned>` means the key was absent or that no successful "
        "precomputed response was available. Zero and false remain reported values. "
        "Meanings explicitly marked 'interpretation from key' are inferred; the tutorial does not document every field or category scale.",
        "Parcel-level attributes cover the Vexcel parcel, which need not match a CoStar community boundary. "
        "The 80 m imagery square may cover only part of a large community. A tree-cover percentage is not total green cover, "
        "and a detected vehicle count cannot validate parking-space capacity.", "",
        "| Exact response field | " + " | ".join(str(result["property"]["id"]) for result in results) + " | Meaning |",
        "|---|" + "---|" * len(results) + "---|",
    ])
    successful = [r["response"] for r in results if r.get("response") is not None]
    paths = set().union(*(field_paths(response) for response in successful)) if successful else set()
    for path in sorted(paths, key=path_label):
        values = [field_value(result["response"], path) if result.get("response") is not None else ABSENT for result in results]
        lines.append(f"| `{path_label(path)}` | " + " | ".join(markdown_value(value) for value in values) + f" | {field_meaning(path)} |")
    if not paths:
        lines.append("| No successful precomputed response | " + " | ".join("unavailable" for _ in results) + " | No observed field values can be listed. |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path.name, flush=True)


def run_probe(
    client: CachedVexcelClient, data_path: Path = DATA_PATH,
    samples_dir: Path = SAMPLES_DIR, report_path: Path = REPORT_PATH,
) -> bool:
    """Fetch the two requested artifacts per property and always document outcomes."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_ok = True
    for prop in select_properties(data_path):
        result = {"property": prop}
        property_params, image_body = request_payloads(prop["latitude"], prop["longitude"])
        for kind, method, endpoint, payload, timeout in (
            ("property", "GET", "property/extract", property_params, 120),
            ("image", "POST", "ortho/extract", image_body, 300),
        ):
            try:
                response = client.request(method, endpoint, payload, timeout)
                result[f"{kind}_status"] = response.status_code
                if not 200 <= response.status_code < 300:
                    all_ok = False
                    continue
                if kind == "property":
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
                        raise ValueError("invalid_property_response")
                    destination = samples_dir / f"vexcel_property_{prop['id']}.json"
                    destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    result.update(response=data, json_file=destination.name)
                else:
                    # JPEG signatures prevent saving an HTTP 200 JSON error as a .jpg.
                    if not response.content.startswith(b"\xff\xd8\xff"):
                        raise ValueError("invalid_jpeg_response")
                    destination = samples_dir / f"vexcel_{prop['id']}.jpg"
                    destination.write_bytes(response.content)
                    result["image_file"] = destination.name
                print(destination.name, flush=True)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                # Never print exception text: an upstream body/error might contain credentials.
                if isinstance(error, RuntimeError) and str(error) in {"cache_miss", "missing_credential"}:
                    result[f"{kind}_error"] = str(error)
                elif isinstance(error, requests.RequestException) and not isinstance(error, ValueError):
                    result[f"{kind}_error"] = "transport_error"
                else:
                    result[f"{kind}_error"] = "invalid_response"
                all_ok = False
        results.append(result)
    write_field_report(results, report_path)
    return all_ok


def main() -> int:
    """Load .env credentials/settings and run the probe with status-only output."""
    load_dotenv(PROJECT_ROOT / ".env")
    client = CachedVexcelClient(
        os.getenv("VEXCEL_API_KEY"),
        cache_only=os.getenv("USE_CACHE_ONLY", "false").strip().lower() == "true",
    )
    return 0 if run_probe(client) else 1


if __name__ == "__main__":
    raise SystemExit(main())
