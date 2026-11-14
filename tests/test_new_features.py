"""Tests for heat/cool roofs, lot fullness, Empty Lot Planner and change over time (no network, no model)."""

import base64
import json
from unittest.mock import Mock

import pytest
from PIL import Image

from backend import llm
from backend.schemas import Building, Detection, Measurement, VexcelPropertyInfo
from backend.tools import change, heat, lot, roof_facts, solar, vision


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(solar.config, "NREL_API_KEY", None)
    monkeypatch.setattr(solar.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lot.config, "USE_CACHE_ONLY", False)


INFO = VexcelPropertyInfo(id="v", attributes={
    "parcel-tree-cover.percent": 3.0, "hardscapes.percent": 60.0, "hardscapes.area": 20000.0,
    "vehicles.count": 12, "metadata.ortho-date-estimate": "2025-03-11T00:00:00"},
    structures=[{"primary-structure": True, "footprint.area": 45000.0, "roof-characteristics.material.class": "membrane",
                 "roof-characteristics.shape.value": "flat", "roof-objects.ac-unit.count": 10,
                 "roof-objects.roof-vent.count": 20, "roof-objects.skylight.count": 0, "roof-objects.solar.value": False},
                {"footprint.area": 5000.0, "roof-objects.ac-unit.count": 2}])


def test_roof_facts_reads_vexcel_keys() -> None:
    f = roof_facts.roof_facts(INFO)
    assert f["footprint_sq_ft"] == 50000 and f["primary_footprint_sq_ft"] == 45000
    assert f["ac_units"] == 12 and f["roof_vents"] == 20 and f["existing_solar"] is False
    assert f["parcel_tree_cover_percent"] == 3.0 and f["ortho_date"] == "2025-03-11" and f["vehicles_count"] == 12
    assert roof_facts.obstruction_level(f) == "few"  # 32 objects / 50k sq ft = 6.4 per 10k
    assert roof_facts.roof_facts(None)["footprint_sq_ft"] is None


def test_heat_rank_and_cool_roof() -> None:
    facts = roof_facts.roof_facts(INFO)
    b1, b2, b3 = (Building(id=i, market_id=14) for i in (1, 2, 3))
    ms = [Measurement(id="a", property_id=1, observations={"roof_color": "dark"}),
          Measurement(id="b", property_id=2, roof_area_sq_ft=10000, green_cover_percent=50, observations={"roof_color": "light"}),
          Measurement(id="c", property_id=3, observations={})]
    rows = heat.rank_heat([b1, b2, b3], ms, {1: facts})
    assert [r["property_id"] for r in rows] == [1, 2, 3]
    assert rows[0]["heat_score"] == 100 and rows[0]["roof_source"] == "Vexcel footprint"
    assert rows[2]["heat_score"] is None
    cr = rows[0]["cool_roof"]
    assert cr["cost_usd"] == round(50000 * 1.125) and cr["annual_kwh_saved"] == 15000
    assert cr["payback_years"] == round(50000 * 1.125 / (15000 * 0.09), 1)
    assert rows[0]["summary"].startswith("Dark roof, 4,645 m², 3% tree cover.")
    assert cr["payback_if_recoating_years"] == round(50000 * 0.2 / (15000 * 0.09), 1)
    assert "re-coating anyway" in rows[0]["summary"]


def test_compare_retrofits_sorted_by_payback() -> None:
    facts = roof_facts.roof_facts(INFO)
    opts = heat.compare_retrofits(Building(id=1, market_id=20, latitude=34, longitude=-118),
                                  Measurement(id="a", property_id=1, observations={"roof_color": "dark"}), facts,
                                  carport={"payback_years": 9.0, "cost_usd": 1, "annual_savings_usd": 1, "capacity_kw": 70, "spaces_covered": 20})
    names = [o["option"] for o in opts]
    assert set(names) == {"Rooftop solar", "Solar carport", "Cool roof coating"}
    paybacks = [o["payback_years"] for o in opts]
    assert paybacks == sorted(paybacks)


def test_lot_fullness_flags_leased_but_empty() -> None:
    b = Building(id=7, vacancy_percent=6, property_type=5)
    m = Measurement(id="m", property_id=7, parking_space_count=172, observations={"cars_seen": 31})
    r = lot.lot_fullness(b, m, capture_date="2024-03-12")
    assert r["lot_fullness_percent"] == 18.0 and r["flag"] == "leased but looks empty"
    assert r["summary"] == "Reported: 94% leased. Lot on Tue Mar 12 2024: 18% full (31 of 172 spaces)."
    weekend = lot.lot_fullness(b, m, capture_date="2024-03-10")
    assert weekend["capture_weekday"] is False and weekend["flag"] == "weekend photo, not judged"
    holiday = lot.lot_fullness(b, m, capture_date="2026-01-02")
    assert any("holiday" in c for c in holiday["caveats"])


def test_lot_fullness_guards_bad_space_counts() -> None:
    b = Building(id=7, vacancy_percent=1, surface_parking_spaces=80)
    no_lot = lot.lot_fullness(b, Measurement(id="m", property_id=7, parking_space_count=0, observations={"cars_seen": 0}))
    assert no_lot["lot_fullness_percent"] is None and no_lot["flag"] == "unknown"
    tiny = lot.lot_fullness(Building(id=8), Measurement(id="m", property_id=8, parking_space_count=1, observations={"cars_seen": 2}))
    assert tiny["lot_fullness_percent"] is None
    mixed = lot.lot_fullness(Building(id=9, surface_parking_spaces=350),
                             Measurement(id="m", property_id=9, parking_space_count=1, observations={"cars_seen": 2}))
    assert mixed["lot_fullness_percent"] is None  # never mix an 80 m photo with whole-property CoStar spaces
    capped = lot.lot_fullness(Building(id=9, surface_parking_spaces=10), Measurement(id="m", property_id=9),
                              {"vehicles_count": 8})
    assert capped["spaces_source"] == "CoStar surface spaces" and capped["lot_fullness_percent"] == 80.0
    assert lot.plan_empty_lot(b, no_lot)["options"] == []


def test_lot_fullness_falls_back_to_costar_and_vexcel() -> None:
    b = Building(id=7, vacancy_percent=10, surface_parking_spaces=40)
    r = lot.lot_fullness(b, Measurement(id="m", property_id=7), {"vehicles_count": 30, "ortho_date": "2025-03-11"})
    assert r["spaces_source"] == "CoStar surface spaces" and r["cars_source"] == "Vexcel vehicle count"
    assert r["lot_fullness_percent"] == 75.0 and r["flag"] == "matches"


def test_fullness_vs_vacancy_needs_five() -> None:
    rows = [{"lot_fullness_percent": f, "reported_occupancy_percent": o, "capture_weekday": True}
            for f, o in [(10, 90), (20, 92), (40, 95), (50, 97), (60, 99), (70, 98)]]
    out = lot.fullness_vs_vacancy(rows)
    assert out["n"] == 6 and out["pearson_r"] > 0.8
    assert lot.fullness_vs_vacancy(rows[:3])["n"] == 3


def test_rule_check_la_near_rail_and_houston_downtown() -> None:
    la = lot.parking_rule_check(Building(id=1, market_id=20, latitude=34.0561, longitude=-118.2365))  # Union Station
    assert la["can_remove_spaces"] and "AB 2097" in la["reason"]
    hou = lot.parking_rule_check(Building(id=2, market_id=14, latitude=29.758, longitude=-95.365))
    assert hou["can_remove_spaces"] and "Downtown" in hou["reason"]
    far = lot.parking_rule_check(Building(id=3, market_id=14, latitude=29.95, longitude=-95.65))
    assert not far["can_remove_spaces"]


def test_plan_empty_lot_math_and_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lot.llm, "ask_text", Mock(side_effect=RuntimeError("down")))
    b = Building(id=9, market_id=14, latitude=29.95, longitude=-95.65)
    full = {"spaces": 200, "cars": 40, "lot_fullness_percent": 20.0, "capture_date": "2025-03-11"}
    plan = lot.plan_empty_lot(b, full)
    assert plan["keep_spaces"] == 50 and plan["spare_spaces"] == 150 and plan["spare_land_m2"] == 4500
    names = [o["option"] for o in plan["options"]]
    assert "Build on it" not in names and names[0] == "Rent the spaces"
    rent = plan["options"][0]
    assert rent["annual_income_usd"] == round(75 * 60 * 12 * 0.8)
    assert plan["recommendation"].startswith("This lot was about 20% full on 2025-03-11. Keep 50 spaces")


def test_recommendation_uses_model_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lot.llm, "ask_text", Mock(return_value="Rent 100 spare spaces and add a solar carport (estimated)."))
    plan = lot.plan_empty_lot(Building(id=9, market_id=20, latitude=34.0561, longitude=-118.2365),
                              {"spaces": 200, "cars": 40, "lot_fullness_percent": 20.0})
    assert plan["recommendation"].startswith("Rent 100") and "Build on it" in [o["option"] for o in plan["options"]]


def test_vision_reads_car_boxes_from_detection_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img, format="JPEG")
    run = {"detections": [Detection(id="1", label="parking space", bbox=(0, 0, 10, 10)).model_dump(),
                          Detection(id="2", label="car", bbox=(20, 20, 30, 30), confidence=0.8).model_dump(),
                          Detection(id="3", label="car", bbox=(40, 40, 50, 50)).model_dump()]}
    reply = {"confirmed_box_ids": [1], "parking_space_count": 12, "confirmed_car_ids": [1, 2], "car_count": 2,
             "roof_color": "dark", "confidence": 0.7}
    ask = Mock(return_value=json.dumps(reply))
    monkeypatch.setattr(vision.llm, "ask_with_image", ask)
    m = vision.inspect(str(img), 5, run)
    assert m.parking_space_count == 12 and m.observations["cars_seen"] == 2  # Claude's stall count, not box count
    assert m.observations["roof_color"] == "dark" and m.observations["lot_fullness_percent"] == 16.7
    assert [d.label for d in m.detections].count("car") == 2
    assert "car boxes" in ask.call_args.args[0]


def test_change_pick_dates_and_summary() -> None:
    cols = [{"collection": "a", "layer": "urban", "date": "2025-03-01"}, {"collection": "b", "layer": "wide-area", "date": "2026-01-01"},
            {"collection": "c", "layer": "urban-r", "date": "2023-02-01"}, {"collection": "d", "layer": "urban", "date": "2024-11-01"}]
    before, after = change.pick_dates(cols)
    assert before["date"] == "2023-02-01" and after["date"] == "2025-03-01"
    assert change.pick_dates(cols[:1]) is None
    rows = [{"before_date": "2023-02-01", "after_date": "2025-03-01", "new_solar": True, "lot_fullness_before": 80, "lot_fullness_after": 40},
            {"before_date": "2023-02-01", "after_date": "2025-03-01", "new_solar": False}, {"error": "x"}]
    assert change.summarize_changes(rows) == "1 of 2 buildings added solar between 2023 and 2025; 0 saw new construction; 1 lots got noticeably emptier."
    assert change.change_sentence(rows[0]) == "2023–2025: added solar. Lot 80% → 40% full."


def test_change_compare_uses_cache_and_validates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(change.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(change.config, "USE_CACHE_ONLY", True)
    assert change.compare_dates(Building(id=1, latitude=1, longitude=1))["error"].startswith("not cached")
    (tmp_path / "change").mkdir()
    (tmp_path / "change" / "1.json").write_text(json.dumps({"property_id": 1, "summary": "cached"}))
    assert change.compare_dates(Building(id=1, latitude=1, longitude=1))["summary"] == "cached"
    with pytest.raises(Exception):
        change.parse_change('{"new_solar": true}')
    ok = change.parse_change('Sure {"new_solar": true, // yes\n "new_construction": false, "demolition": false, "roof_changed": false, "confidence": 0.6}')
    assert ok.new_solar is True


def test_ask_with_images_sends_all_images(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    for i in range(2):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (2, 2), "red").save(p, format="JPEG")
        paths.append(str(p))
    model = Mock()
    model.invoke.return_value = Mock(text="ok")
    factory = Mock(return_value=model)
    monkeypatch.setattr(llm, "get_chat_model", factory)
    assert llm.ask_with_images("compare", paths) == "ok"
    content = model.invoke.call_args.args[0][0].content
    assert content[0]["text"] == "compare" and len(content) == 3
    assert base64.b64decode(content[2]["base64"]) == open(paths[1], "rb").read()
    factory.assert_called_once_with("vision")
