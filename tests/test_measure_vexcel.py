"""Tests for scripts/measure_vexcel.py: value extraction, unknown handling, merging."""

from backend.schemas import Measurement
from scripts import measure_vexcel as ms


def test_parcel_values_reads_vexcel_fields() -> None:
    props = {
        "parcel-tree-cover.percent": 14.0,
        "hardscapes.percent": 21.0,
        "structures": [{"footprint.area": 1000.0}, {"footprint.area": 500.5}, {"footprint.area": None}],
        "pool.detected": True,
    }
    v = ms.parcel_values(props)
    assert v["green_cover_percent"] == 14.0
    assert v["paved_surface_percent"] == 21.0
    assert v["roof_area_sq_ft"] == 1500.5
    assert v["structure_count"] == 2


def test_parcel_values_unknown_stays_none() -> None:
    v = ms.parcel_values({"parcel-tree-cover.percent": None, "hardscapes.percent": 150})
    assert v["green_cover_percent"] is None
    assert v["paved_surface_percent"] is None  # out of range -> unknown
    assert v["roof_area_sq_ft"] is None


def _record(basis: str) -> dict:
    return {"id": 7, "basis": basis, "green_cover_percent": 12.0,
            "paved_surface_percent": 8.0, "roof_area_sq_ft": 900.0}


def test_matched_parcel_values_are_used() -> None:
    m = ms.to_measurement(_record("parcel matching CoStar lotSize"), None)
    assert (m.id, m.property_id, m.green_cover_percent, m.roof_area_sq_ft) == ("m-7", 7, 12.0, 900.0)
    assert m.observations["source"] == ms.SOURCE


def test_unmatched_parcel_values_become_unknown() -> None:
    m = ms.to_measurement(_record("parcel at CoStar point (no parcel matched lotSize)"), None)
    assert m.green_cover_percent is None and m.paved_surface_percent is None and m.roof_area_sq_ft is None
    assert m.observations["unmatched_parcel_values"]["green_cover_percent"] == 12.0


def test_sky_inspector_result_is_kept_as_second_opinion() -> None:
    claude = Measurement(id="m-7", property_id=7, green_cover_percent=30.0, parking_space_count=40,
                         observations={"notes": "from vision"})
    m = ms.to_measurement(_record("parcel matching CoStar lotSize"), claude)
    assert m.green_cover_percent == 12.0  # Vexcel wins
    assert m.parking_space_count == 40  # Claude's parking count stays
    assert m.observations["sky_inspector"]["green_cover_percent"] == 30.0
    assert m.observations["notes"] == "from vision"


def test_detections_are_attached() -> None:
    rec = {**_record("parcel matching CoStar lotSize"), "detection": {
        "car_count": 1, "image_path": "a.jpg", "annotated_image_path": "a_boxes.jpg",
        "detections": [{"id": "det-7-1", "label": "car", "property_id": 7, "bbox": [1, 2, 3, 4]}]}}
    m = ms.to_measurement(rec, None)
    assert m.detections[0].label == "car" and m.detections[0].bbox == (1, 2, 3, 4)
    assert m.observations["car_count"] == 1
    assert m.observations["annotated_image_path"] == "a_boxes.jpg"
