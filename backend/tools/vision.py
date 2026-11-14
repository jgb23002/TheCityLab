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
about {extent_m:.0f} m across ({m_per_px:.3f} m per pixel). The gold/yellow square
is an ~80 m box centred on the property's address point, not its exact boundary.
Measure the property at the centre of the square: its buildings and the parking
areas that serve them, inside the square.

Parking spaces = parking capacity of open-air (surface) lots, not covered garages or
carports. Count striped stalls where stripes are visible. Where stripes are faded or
absent, estimate how many cars the paved parking area holds at a standard stall of
2.7 m x 5.5 m with drive aisles, and never return fewer than the cars parked there.
Do not count street parking or driveways of neighbouring lots.

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
  "roof_color": "dark", "medium", "light", "white" or null,
  "car_count": integer or null,
  "confirmed_car_ids": [integers],
  "confidence": number 0-1,
  "notes": "one short sentence about anything unclear"
}}
Meanings:
- parking_space_count: surface parking capacity as defined above.
- confirmed_box_ids: ids of numbered boxes that really are parking spaces of this property.
- extra_parking_boxes: pixel boxes for clear spaces the boxes missed (at most 40).
- roof_fraction_of_image: share of the WHOLE image covered by the target building's roof.
- green_cover_percent / paved_surface_percent: trees+grass / asphalt+concrete as % of the gold square.
- rooftop_obstructions: HVAC units, vents, skylights on the roof.
- roof_color: the main roof's colour as seen (dark = black/dark grey/brown, white = bright reflective).
- car_count: cars and trucks parked in this property's surface lots right now (not on public streets).
- confirmed_car_ids: ids of numbered car boxes that really are parked cars of this property.
- confidence: how sure you are overall.
Use null when you cannot tell. Never guess a number you cannot see."""

BOX_SECTION_WITH_BOXES = """Vexcel's detector proposed these parking-space boxes, as
[id, left, top, right, bottom] in pixels:
{boxes}
These boxes are hints: one box may outline a whole lot or row rather than a single stall.
Confirm which boxes fall on this property's parking, add clear spaces it missed, and set
parking_space_count to the number of individual stalls you count."""

BOX_SECTION_NO_BOXES = "No detector boxes are available; count parking spaces yourself."

CAR_SECTION = """Vexcel's detector proposed these car boxes, as [id, left, top, right, bottom] in pixels:
{boxes}
Confirm which are parked cars of this property and set car_count = confirmed + any clear cars it missed."""

CAR_SECTION_NONE = "No car boxes are available; count parked cars yourself."

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
    roof_color: Literal["dark", "medium", "light", "white"] | None = None
    car_count: int | None = Field(default=None, ge=0)
    confirmed_car_ids: list[int] = Field(default_factory=list)
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


def _vexcel_px_boxes(vexcel_detections: Any, label: str = "parking space") -> list[dict[str, Any]]:
    """Pixel boxes for one label from ``vexcel_detect.detect`` (DetectionRun) or the older dict.

    Returns ``[{"bbox_px": [l, t, r, b], "score": float | None}]``.
    """
    if not vexcel_detections:
        return []
    if hasattr(vexcel_detections, "model_dump"):
        vexcel_detections = vexcel_detections.model_dump(mode="json")
    boxes: list[dict[str, Any]] = []
    for det in vexcel_detections.get("detections") or []:  # DetectionRun
        if isinstance(det, dict) and det.get("label") == label and isinstance(det.get("bbox"), (list, tuple)) and len(det["bbox"]) == 4:
            boxes.append({"bbox_px": list(det["bbox"]), "score": det.get("confidence")})
    for box in vexcel_detections.get("boxes") or []:  # older dict format
        px = box.get("bbox_px") if isinstance(box, dict) else None
        if isinstance(px, list) and len(px) == 4 and box.get("label", label) == label:
            boxes.append({"bbox_px": px, "score": box.get("score")})
    return boxes[:MAX_BOXES_IN_PROMPT]


def _listed(boxes: list[dict[str, Any]]) -> str:
    """Number boxes from 1 as JSON lines [id, l, t, r, b]."""
    return "\n".join(json.dumps([i, *[round(float(v)) for v in b["bbox_px"]]]) for i, b in enumerate(boxes, start=1))


def build_prompt(
    width: int, height: int, extent_m: float, boxes: list[dict[str, Any]],
    car_boxes: list[dict[str, Any]] | None = None,
) -> str:
    """Fill the vision prompt with the image scale and any numbered Vexcel boxes."""
    section = BOX_SECTION_WITH_BOXES.format(boxes=_listed(boxes)) if boxes else BOX_SECTION_NO_BOXES
    section += "\n\n" + (CAR_SECTION.format(boxes=_listed(car_boxes)) if car_boxes else CAR_SECTION_NONE)
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
    vexcel_detections: Any = None,
    image_extent_m: float = DEFAULT_IMAGE_EXTENT_M,
) -> Measurement:
    """Measure one building from its aerial JPEG, checking Vexcel's boxes when given.

    Args:
        image_path: Local aerial JPEG (from vexcel.get_aerial).
        property_id: CoStar property id.
        vexcel_detections: Optional ``vexcel_detect.detect`` result run with prompts
            ``["parking space", "car"]``; Claude confirms both kinds of box.
        image_extent_m: Ground width of the image in metres, used for roof area.

    Returns:
        A Measurement. On failure every value is null and ``observations.error`` says why.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    boxes = _vexcel_px_boxes(vexcel_detections, "parking space")
    car_boxes = _vexcel_px_boxes(vexcel_detections, "car")
    m_per_px = image_extent_m / width
    observations: dict[str, Any] = {
        "basis": f"aerial image ~{image_extent_m:.0f} m across around the property point; "
        "percentages are of the ~80 m square as judged by the model, not the legal parcel",
        "image_width_px": width,
        "image_height_px": height,
        "meters_per_pixel": round(m_per_px, 4),
        "vexcel_box_count": len(boxes),
        "vexcel_car_box_count": len(car_boxes),
        "vision_model": llm.vision_model_id() or "LLM_MODEL",
    }
    base = Measurement(id=f"m-{property_id}", property_id=property_id, aerial_image_id=Path(image_path).stem)

    reply, error = _ask_json(build_prompt(width, height, image_extent_m, boxes, car_boxes), image_path)
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

    cars_confirmed = sorted({i for i in reply.confirmed_car_ids if 1 <= i <= len(car_boxes)})
    for i in cars_confirmed:
        bbox = _clamp_box(car_boxes[i - 1]["bbox_px"], width, height)
        if bbox:
            score = car_boxes[i - 1].get("score")
            detections.append(Detection(
                id=f"{property_id}-car{i}", label="car", property_id=property_id, bbox=bbox,
                confidence=float(score) if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
            ))
    cars = reply.car_count
    if car_boxes and cars is None:
        cars = len(cars_confirmed)

    # Vexcel "parking space" boxes often outline whole lots, so Claude's stall count wins.
    if reply.parking_space_count is not None:
        parking = reply.parking_space_count
    else:
        parking = (len(confirmed) + len(extra)) or None

    roof_sq_ft = None
    if reply.roof_fraction_of_image is not None:
        image_area_m2 = (width * m_per_px) * (height * m_per_px)
        roof_sq_ft = round(reply.roof_fraction_of_image * image_area_m2 * SQ_FT_PER_M2)

    observations.update({
        "solar_panels_present": reply.solar_panels_present,
        "rooftop_obstructions": reply.rooftop_obstructions,
        "roof_color": reply.roof_color,
        "cars_seen": cars,
        "vexcel_cars_confirmed": len(cars_confirmed),
        "lot_fullness_percent": round(100 * cars / parking, 1) if cars is not None and parking and parking >= cars else None,
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
