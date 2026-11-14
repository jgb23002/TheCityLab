"""Sky Inspector: Claude checks Vexcel detections on one aerial image and measures it.

``measure_building`` is the frozen contract. ``inspect`` is the same call but can
also take the output of ``vexcel.detect`` so Claude confirms Vexcel's parking
boxes instead of counting from scratch.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from backend import llm
from backend.schemas import Detection, Measurement

log = logging.getLogger(__name__)

# get_aerial asks Vexcel for an ~80 m square plus a 10 m buffer on every side.
DEFAULT_IMAGE_EXTENT_M = 100.0
SQ_FT_PER_M2 = 10.7639
MAX_BOXES_IN_PROMPT = 150
MAX_EXTRA_BOXES = 40

VISION_PROMPT = """You are the Sky Inspector for a building research study.
The image is a top-down aerial photo, north up, {width}x{height} pixels,
about {extent_m:.0f} m across ({m_per_px:.3f} m per pixel). The target property
is outlined in gold/yellow. Measure ONLY what is inside that outline.

{box_section}

Return JSON only, no prose, no code fences, with exactly these keys:
{{
  "parking_space_count": integer or null,
  "confirmed_box_ids": [integers],
  "extra_parking_boxes": [[left, top, right, bottom], ...],
  "roof_fraction_of_image": number 0-1 or null,
  "green_cover_percent": number 0-100 or null,
  "paved_surface_percent": number 0-100 or null,
  "solar_panels_present": true, false or null,
  "rooftop_obstructions": "none", "few", "many" or null,
  "confidence": number 0-1,
  "notes": "one short sentence about anything unclear"
}}
Meanings:
- parking_space_count: painted parking spaces (not cars) inside the outline.
- confirmed_box_ids: ids of numbered boxes that really are parking spaces inside the outline.
- extra_parking_boxes: pixel boxes for clear spaces the boxes missed (at most 40).
- roof_fraction_of_image: share of the WHOLE image covered by the target building's roof.
- green_cover_percent / paved_surface_percent: trees+grass / asphalt+concrete as % of the outlined area.
- rooftop_obstructions: HVAC units, vents, skylights on the roof.
- confidence: how sure you are overall.
Use null when you cannot tell. Never guess a number you cannot see."""

BOX_SECTION_WITH_BOXES = """Vexcel's detector proposed these parking-space boxes, as
[id, left, top, right, bottom] in pixels:
{boxes}
Confirm which ones are real parking spaces inside the outline, add clear spaces it missed,
and set parking_space_count = confirmed + extra."""

BOX_SECTION_NO_BOXES = "No detector boxes are available; count parking spaces yourself."

RETRY_SUFFIX = "\n\nYour last answer was not valid JSON with the required keys. Reply with the JSON object only."


class VisionReply(BaseModel):
    """The strict JSON shape the vision model must return."""

    parking_space_count: int | None = Field(default=None, ge=0)
    confirmed_box_ids: list[int] = Field(default_factory=list)
    extra_parking_boxes: list[list[float]] = Field(default_factory=list)
    roof_fraction_of_image: float | None = Field(default=None, ge=0, le=1)
    green_cover_percent: float | None = Field(default=None, ge=0, le=100)
    paved_surface_percent: float | None = Field(default=None, ge=0, le=100)
    solar_panels_present: bool | None = None
    rooftop_obstructions: Literal["none", "few", "many"] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None


def parse_reply(text: str) -> VisionReply:
    """Pull the first JSON object out of a model reply and validate it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in reply")
    return VisionReply.model_validate(json.loads(match.group(0)))


def _clamp_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Return a [left, top, right, bottom] box clipped to the image, or None if invalid."""
    if len(box) != 4:
        return None
    left, top, right, bottom = (float(v) for v in box)
    left, right = sorted((max(0.0, min(left, width)), max(0.0, min(right, width))))
    top, bottom = sorted((max(0.0, min(top, height)), max(0.0, min(bottom, height))))
    if right - left < 1 or bottom - top < 1:
        return None
    return (left, top, right, bottom)


def _vexcel_px_boxes(vexcel_detections: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Keep only Vexcel boxes that have a pixel box (``bbox_px`` from vexcel.detect)."""
    if not vexcel_detections:
        return []
    boxes = []
    for box in vexcel_detections.get("boxes") or []:
        px = box.get("bbox_px") if isinstance(box, dict) else None
        if isinstance(px, list) and len(px) == 4:
            boxes.append(box)
    return boxes[:MAX_BOXES_IN_PROMPT]


def build_prompt(width: int, height: int, extent_m: float, boxes: list[dict[str, Any]]) -> str:
    """Fill the vision prompt with the image scale and any numbered Vexcel boxes."""
    if boxes:
        listed = "\n".join(
            json.dumps([i, *[round(float(v)) for v in b["bbox_px"]]]) for i, b in enumerate(boxes, start=1)
        )
        section = BOX_SECTION_WITH_BOXES.format(boxes=listed)
    else:
        section = BOX_SECTION_NO_BOXES
    return VISION_PROMPT.format(
        width=width, height=height, extent_m=extent_m, m_per_px=extent_m / width, box_section=section
    )


def _ask_json(prompt: str, image_path: str) -> tuple[VisionReply | None, str | None]:
    """Ask the vision model, retrying once on bad JSON. Returns (reply, error)."""
    error = None
    for attempt in range(2):
        try:
            text = llm.ask_with_image(prompt + (RETRY_SUFFIX if attempt else ""), image_path, llm.vision_model_id())
        except Exception as exc:  # provider errors; never echo their bodies
            log.warning("Vision call failed for %s: %s", image_path, type(exc).__name__)
            return None, f"vision call failed: {type(exc).__name__}"
        try:
            return parse_reply(text), None
        except (ValueError, ValidationError) as exc:
            error = f"bad vision JSON: {type(exc).__name__}"
            log.warning("Vision reply for %s failed validation (attempt %d)", image_path, attempt + 1)
    return None, error


def inspect(
    image_path: str,
    property_id: int,
    vexcel_detections: dict[str, Any] | None = None,
    image_extent_m: float = DEFAULT_IMAGE_EXTENT_M,
) -> Measurement:
    """Measure one building from its aerial JPEG, checking Vexcel's boxes when given.

    Args:
        image_path: Local aerial JPEG (from vexcel.get_aerial).
        property_id: CoStar property id.
        vexcel_detections: Optional output of ``vexcel.detect`` (uses its ``boxes[].bbox_px``).
        image_extent_m: Ground width of the image in metres, used for roof area.

    Returns:
        A Measurement. On failure every value is null and ``observations.error`` says why.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    boxes = _vexcel_px_boxes(vexcel_detections)
    m_per_px = image_extent_m / width
    observations: dict[str, Any] = {
        "basis": f"aerial image ~{image_extent_m:.0f} m across around the property point; "
        "percentages are of the outlined area as judged by the model, not the legal parcel",
        "image_width_px": width,
        "image_height_px": height,
        "meters_per_pixel": round(m_per_px, 4),
        "vexcel_box_count": len(boxes),
        "vision_model": llm.vision_model_id() or "LLM_MODEL",
    }
    base = Measurement(id=f"m-{property_id}", property_id=property_id, aerial_image_id=Path(image_path).stem)

    reply, error = _ask_json(build_prompt(width, height, image_extent_m, boxes), image_path)
    if reply is None:
        observations["error"] = error
        return base.model_copy(update={"observations": observations})

    detections: list[Detection] = []
    confirmed = sorted({i for i in reply.confirmed_box_ids if 1 <= i <= len(boxes)})
    for i in confirmed:
        b = boxes[i - 1]
        bbox = _clamp_box(b["bbox_px"], width, height)
        if bbox:
            score = b.get("score")
            detections.append(Detection(
                id=f"{property_id}-v{i}", label="parking space", property_id=property_id, bbox=bbox,
                confidence=float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
            ))
    extra = [bb for bb in (_clamp_box(b, width, height) for b in reply.extra_parking_boxes[:MAX_EXTRA_BOXES]) if bb]
    for j, bbox in enumerate(extra, start=1):
        detections.append(Detection(id=f"{property_id}-c{j}", label="parking space", property_id=property_id, bbox=bbox))

    if boxes:
        parking = len(confirmed) + len(extra)
    else:
        parking = reply.parking_space_count if reply.parking_space_count is not None else (len(extra) or None)

    roof_sq_ft = None
    if reply.roof_fraction_of_image is not None:
        image_area_m2 = (width * m_per_px) * (height * m_per_px)
        roof_sq_ft = round(reply.roof_fraction_of_image * image_area_m2 * SQ_FT_PER_M2)

    observations.update({
        "solar_panels_present": reply.solar_panels_present,
        "rooftop_obstructions": reply.rooftop_obstructions,
        "roof_fraction_of_image": reply.roof_fraction_of_image,
        "vexcel_boxes_confirmed": len(confirmed),
        "parking_spaces_added_by_model": len(extra),
        "model_parking_count": reply.parking_space_count,
        "notes": reply.notes,
    })
    return base.model_copy(update={
        "roof_area_sq_ft": roof_sq_ft,
        "confidence": reply.confidence,
        "green_cover_percent": reply.green_cover_percent,
        "paved_surface_percent": reply.paved_surface_percent,
        "parking_space_count": parking,
        "detections": detections,
        "observations": observations,
    })


def measure_building(image_path: str, property_id: int) -> Measurement:
    """Measure the building from a cached aerial image with confidence."""
    return inspect(image_path, property_id)
