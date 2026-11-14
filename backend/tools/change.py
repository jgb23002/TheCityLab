"""Change over time: compare an old and a new Vexcel photo of the same building.

Uses ``vexcel_detect.list_collections`` for the capture dates at a point, saves one JPEG
per chosen date, and sends both to the vision model in one message. Results are cached
in data/cache/change/ so the demo replays offline. Seasons, shadows and sun angle can
fool the model, and the two photos may not line up exactly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from shapely.geometry import box

from backend import config, llm
from backend.schemas import Building

log = logging.getLogger(__name__)

SHARP_LAYERS = ("urban", "urban-r")
MIN_YEARS_APART = 1

CHANGE_PROMPT = """Two top-down aerial photos of the same ~80 m square, north up.
Image 1 was taken on {before}; image 2 on {after}. The building at the centre is the subject.
Compare them and return JSON only, no prose:
{{
  "new_solar": true/false,          // solar panels on the subject roof in image 2 but not image 1
  "new_construction": true/false,   // a new building or addition on the subject site
  "demolition": true/false,         // a structure on the subject site is gone
  "roof_changed": true/false,       // roof re-covered, recoloured or rebuilt
  "lot_fullness_before": 0-100 or null,  // % of the subject's surface parking spaces with a car, image 1
  "lot_fullness_after": 0-100 or null,   // same for image 2
  "confidence": 0-1,
  "notes": "one sentence on the most important change, or 'no clear change'"
}}
Ignore differences caused only by season, shadows, sun angle or slight misalignment."""


class ChangeReply(BaseModel):
    """Strict JSON shape for a two-date comparison."""

    new_solar: bool
    new_construction: bool
    demolition: bool
    roof_changed: bool
    lot_fullness_before: float | None = Field(default=None, ge=0, le=100)
    lot_fullness_after: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    notes: str | None = None


def pick_dates(collections: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]] | None:
    """Newest sharp capture and the oldest one at least MIN_YEARS_APART earlier."""
    sharp = sorted((c for c in collections if c.get("layer") in SHARP_LAYERS), key=lambda c: c["date"])
    if len(sharp) < 2:
        return None
    after = sharp[-1]
    before = next((c for c in sharp if int(after["date"][:4]) - int(c["date"][:4]) >= MIN_YEARS_APART), None)
    return (before, after) if before else None


def _image_for(lat: float, lon: float, property_id: int, collection: dict[str, str]) -> str:
    """Save (or reuse) the ~80 m JPEG for one capture date."""
    from backend.tools import vexcel

    path = config.CACHE_DIR / "vexcel" / "dates" / f"property_{property_id}_{collection['date']}.jpg"
    if path.exists():
        return str(path)
    area = box(*vexcel._square_bounds(lat, lon, half_m=vexcel.SQUARE_HALF_M)).wkt
    data = vexcel._client.request_bytes("POST", "ortho/extract", {
        "layer": collection["layer"], "collection": collection["collection"], "srid": 4326, "wkt": area,
        "crop": "bounding_box", "buffer-meter": 0, "image-format": "jpeg",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def parse_change(text: str) -> ChangeReply:
    """Pull and validate the JSON object from a model reply (tolerates // comments)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    return ChangeReply.model_validate(json.loads(re.sub(r"//[^\n]*", "", match.group(0))))


def compare_dates(building: Building, refresh: bool = False) -> dict[str, Any]:
    """Before/after comparison for one building; cached; error field instead of raising."""
    cache = config.CACHE_DIR / "change" / f"{building.id}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    base: dict[str, Any] = {"property_id": building.id}
    if building.latitude is None or building.longitude is None:
        return {**base, "error": "no location"}
    if config.USE_CACHE_ONLY:
        return {**base, "error": "not cached (cache-only mode)"}
    try:
        from backend import vexcel_detect

        pair = pick_dates(vexcel_detect.list_collections(building.latitude, building.longitude))
        if not pair:
            return {**base, "error": f"fewer than two sharp captures {MIN_YEARS_APART}+ years apart"}
        before, after = pair
        paths = [_image_for(building.latitude, building.longitude, building.id, c) for c in (before, after)]
    except Exception as exc:
        log.warning("Vexcel dates failed for %s: %s", building.id, type(exc).__name__)
        return {**base, "error": f"imagery unavailable: {type(exc).__name__}"}

    prompt = CHANGE_PROMPT.format(before=before["date"], after=after["date"])
    reply, error = None, None
    for attempt in range(2):
        try:
            reply = parse_change(llm.ask_with_images(prompt + ("\nReply with the JSON object only." if attempt else ""),
                                                     paths, llm.vision_model_id()))
            break
        except (ValueError, ValidationError):
            error = "bad change JSON"
        except Exception as exc:
            error = f"vision call failed: {type(exc).__name__}"
            break
    if reply is None:
        return {**base, "before_date": before["date"], "after_date": after["date"], "error": error}
    result = {**base, "before_date": before["date"], "after_date": after["date"],
              "before_image": paths[0], "after_image": paths[1], **reply.model_dump()}
    result["summary"] = change_sentence(result)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def change_sentence(r: dict[str, Any]) -> str:
    """One line for the building card."""
    things = [label for key, label in (("new_solar", "added solar"), ("new_construction", "new construction"),
                                       ("demolition", "demolition"), ("roof_changed", "roof changed")) if r.get(key)]
    span = f"{r['before_date'][:4]}–{r['after_date'][:4]}"
    text = f"{span}: " + (", ".join(things) if things else "no clear change") + "."
    b, a = r.get("lot_fullness_before"), r.get("lot_fullness_after")
    if b is not None and a is not None:
        text += f" Lot {b:g}% → {a:g}% full."
    return text


def summarize_changes(results: list[dict[str, Any]]) -> str:
    """Brief line, e.g. '6 of 45 buildings added solar between 2024 and 2026; 4 lots got noticeably emptier.'"""
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return "No buildings had two comparable capture dates."
    years = sorted({r["before_date"][:4] for r in ok} | {r["after_date"][:4] for r in ok})
    solar_n = sum(1 for r in ok if r.get("new_solar"))
    built = sum(1 for r in ok if r.get("new_construction"))
    emptier = sum(1 for r in ok if r.get("lot_fullness_before") is not None and r.get("lot_fullness_after") is not None
                  and r["lot_fullness_before"] - r["lot_fullness_after"] >= 20)
    return (f"{solar_n} of {len(ok)} buildings added solar between {years[0]} and {years[-1]}; "
            f"{built} saw new construction; {emptier} lots got noticeably emptier.")
