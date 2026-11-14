"""Sky Inspector tests; the vision model is always mocked."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from backend.tools import vision

GOOD = {
    "parking_space_count": 3, "confirmed_box_ids": [1, 3, 99], "extra_parking_boxes": [[10, 10, 30, 40], [5, 5, 5, 5]],
    "roof_fraction_of_image": 0.25, "green_cover_percent": 12, "paved_surface_percent": 60,
    "solar_panels_present": False, "rooftop_obstructions": "few", "confidence": 0.8, "notes": "clear image",
}
DETECT = {"boxes": [{"bbox_px": [0, 0, 20, 20], "score": 0.9}, {"bbox_px": [20, 20, 40, 40]},
                    {"bbox_px": [50, 50, 70, 90], "score": 0.7}, {"bbox_lonlat": [1, 2, 3, 4]}]}


@pytest.fixture
def image(tmp_path: Path) -> str:
    path = tmp_path / "property_7.jpg"
    Image.new("RGB", (100, 100), "gray").save(path, format="JPEG")
    return str(path)


def test_confirms_vexcel_boxes_and_adds_missed_spaces(image: str, monkeypatch: pytest.MonkeyPatch) -> None:
    ask = Mock(return_value="Here you go:\n" + json.dumps(GOOD))
    monkeypatch.setattr(vision.llm, "ask_with_image", ask)
    m = vision.inspect(image, 7, DETECT)
    assert m.parking_space_count == 3  # boxes 1 and 3 confirmed, one valid extra, 99 ignored
    assert [d.id for d in m.detections] == ["7-v1", "7-v3", "7-c1"]
    assert m.detections[0].confidence == 0.9
    # 100 px over 100 m -> 10,000 m2 image; 25% of it in sq ft
    assert m.roof_area_sq_ft == round(0.25 * 10_000 * vision.SQ_FT_PER_M2)
    assert m.observations["vexcel_box_count"] == 3
    assert "[3, 50, 50, 70, 90]" in ask.call_args.args[0]


def test_retries_once_then_reports_error(image: str, monkeypatch: pytest.MonkeyPatch) -> None:
    ask = Mock(return_value="not json")
    monkeypatch.setattr(vision.llm, "ask_with_image", ask)
    m = vision.measure_building(image, 7)
    assert ask.call_count == 2
    assert m.parking_space_count is None and m.roof_area_sq_ft is None
    assert "bad vision JSON" in m.observations["error"]


def test_provider_error_is_not_echoed(image: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vision.llm, "ask_with_image", Mock(side_effect=RuntimeError("secret-sentinel")))
    m = vision.measure_building(image, 7)
    assert "secret-sentinel" not in json.dumps(m.observations)


def test_without_boxes_uses_model_count(image: str, monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {**GOOD, "confirmed_box_ids": [], "extra_parking_boxes": [], "parking_space_count": 14}
    monkeypatch.setattr(vision.llm, "ask_with_image", Mock(return_value=json.dumps(reply)))
    assert vision.measure_building(image, 7).parking_space_count == 14


def test_out_of_range_values_are_rejected() -> None:
    with pytest.raises(Exception):
        vision.parse_reply(json.dumps({**GOOD, "green_cover_percent": 140}))
