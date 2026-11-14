"""Completion work: Matterport, property lookup, interior + upgrade tools, office sampling,
property flow, plan events, retrofit corrections and calibration. Offline; no keys."""

import socket
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from backend import config
from backend.agent import agent_api, planner, property_lookup, registry, upgrades
from backend.agent.citations import verify_brief
from backend.agent.correction import CORRECTION_HEADING, apply_correction, rescale_solar
from backend.agent.evidence import Ledger
from backend.agent.study_store import StudyRun, calibration_summary, save_study_run
from backend.schemas import Building, Measurement, MeasurementCorrection, OutreachDraft, SolarResult, StudyResult
from backend.tools import costar_data, matterport, solar


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def tools(mode: str, ledger: Ledger, **kw: Any) -> dict:
    return {t.name: t for t in registry.get_tools(mode, ledger, **kw)}


# ---- Matterport ------------------------------------------------------------------------

MODEL = {"id": "abc", "name": "Trade Center", "created": "2026-01-01T00:00:00Z", "visibility": "public",
         "floors": [{"id": "f1", "label": "Floor 1", "dimensions": {"areaFloor": 11650.0, "units": "imperial"}}],
         "rooms": [{"id": "r1", "label": "Warehouse", "tags": ["other"],
                    "dimensions": {"areaFloor": 8273.4, "width": 99.3, "depth": 83.5, "units": "imperial"}},
                   {"id": "r2", "label": "Office", "tags": [], "dimensions": {"height": 9.5, "areaFloor": 300.0}}]}


def test_matterport_summary_reports_only_measured_facts() -> None:
    s = matterport.summarize(MODEL)
    assert s["floor_count"] == 1 and s["scanned_floor_area"] == 11650.0 and s["room_count"] == 2
    assert s["labelled_rooms"] == ["Office", "Warehouse"] and s["measured_ceiling_heights"] == [9.5]
    assert s["units"] == "imperial"


def test_matterport_get_model_uses_basic_auth_caches_and_respects_cache_only(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    monkeypatch.setattr(config, "MP_TOKEN_ID", "id-x")
    monkeypatch.setattr(config, "MP_TOKEN_SECRET", "secret-x")
    post = Mock(return_value=Mock(json=lambda: {"data": {"model": MODEL}}, raise_for_status=lambda: None))
    monkeypatch.setattr(matterport.requests, "post", post)
    first = matterport.get_model("abc")
    assert first.metadata["room_count"] == 2 and first.raw_response is None
    assert post.call_args.kwargs["auth"] == ("id-x", "secret-x")
    matterport.get_model("abc")
    assert post.call_count == 1  # second call replayed from data/cache/matterport
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    with pytest.raises(Exception, match="(?i)cache"):
        matterport.get_model("not-cached")


def test_matterport_errors_are_explicit(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    monkeypatch.setattr(config, "MP_TOKEN_ID", None)
    with pytest.raises(RuntimeError, match="MP_TOKEN_ID"):
        matterport.get_model("abc")


# ---- property lookup ---------------------------------------------------------------------

@pytest.mark.parametrize("query, pid", [("23453", 23453), ("property 23453", 23453), ("Property ID #990005", 990005),
                                        ("Evaluate property 1234567 for solar", 1234567), ("4620 N Braeswood Blvd", None)])
def test_parse_property_id(query: str, pid: int | None) -> None:
    assert property_lookup.parse_property_id(query) == pid


@pytest.mark.parametrize("question, intent", [
    ("Evaluate 4620 N Braeswood Blvd, Houston for upgrades", "property"),
    ("property 23453", "property"),
    ("Which commercial buildings in Los Angeles have the best rooftop solar retrofit potential?", "retrofit"),
    ("Do office buildings with more parking spaces have higher vacancy rates in Houston?", "research"),
])
def test_intent_routing_for_the_demo_scenarios(question: str, intent: str) -> None:
    assert registry.detect_intent(question) == intent


def test_mock_lookup_by_id_and_name_and_unknown() -> None:
    assert property_lookup.lookup("990005", "mock")["building"]["id"] == 990005
    assert property_lookup.lookup("Mock Houston Community 05", "mock")["building"]["id"] == 990005
    with pytest.raises(LookupError, match="No demo property"):
        property_lookup.lookup("12345", "mock")


@pytest.fixture
def fake_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny commercial-property / company-detail / location-occupancy frames."""
    props = pd.DataFrame([{"id": 500, "primaryAddress": {"streetNumber": "12", "streetName": "Main", "streetType": "St",
                                                          "city": "Houston", "state": "TX"},
                           "contacts": [{"contactRole": 7, "locationId": 1}, {"contactRole": 4, "locationId": 2}]},
                          {"id": 501, "primaryAddress": {"streetNumber": "14", "streetName": "Main", "streetType": "St",
                                                          "city": "Houston"}, "contacts": []}])
    companies = pd.DataFrame([{"id": 1, "name": "Acme Holdings"}, {"id": 2, "name": "Acme Management"},
                              {"id": 3, "name": "Corner Cafe"}])
    occupancy = pd.DataFrame([{"propertyId": 500, "tenantLocationId": 3, "toDate": None},
                              {"propertyId": 500, "tenantLocationId": None, "toDate": None},
                              {"propertyId": 500, "tenantLocationId": 3, "toDate": "2020-01-01"}])
    frames = {"commercial-property.jsonl": props, "company-detail.jsonl": companies, "location-occupancy.jsonl": occupancy}
    monkeypatch.setattr(costar_data, "_load_jsonl", lambda name: frames[name])
    monkeypatch.setattr(costar_data, "get_property", lambda id: Building(id=id, name="Main 12", latitude=29.7, longitude=-95.4,
                                                                         primary_address={"streetNumber": "12"}, year_built=1990)
                        if id in (500, 501) else None)
    for fn in (property_lookup._address_index, property_lookup._companies):
        fn.cache_clear()
    yield
    for fn in (property_lookup._address_index, property_lookup._companies):
        fn.cache_clear()


def test_real_lookup_by_id_and_address_with_owners_and_tenants(fake_kit: None) -> None:
    r = property_lookup.lookup("500", "real")
    assert r["owners"] == {"true_owner": ["Acme Holdings"], "recorded_owner": None, "property_manager": ["Acme Management"]}
    assert r["tenants"] == {"current_tenant_count": 2, "tenants": ["Corner Cafe"], "unnamed_tenants": 1}
    assert "max_rent" in r["unavailable"] and "year_built" not in r["unavailable"]
    assert property_lookup.lookup("12 Main St, Houston", "real")["matched_by"] == "address"
    with pytest.raises(LookupError, match="matches 2 properties"):
        property_lookup.lookup("Main St", "real")
    with pytest.raises(LookupError, match="No CoStar property"):
        property_lookup.lookup("99 Elm Rd", "real")


# ---- registry tools: lookup, interior, upgrades, office sampling ------------------------

def test_lookup_tool_feeds_the_inspection_tools() -> None:
    ledger = Ledger()
    t = tools("mock", ledger, intent="property", top_n=1)
    out = t["lookup_property"].invoke({"query": "990005"})
    assert out["building"]["id"] == 990005 and out["evidence_id"] == "E1"
    t["vision_inspect"].invoke({"property_id": 990005})  # uses the looked-up building
    assert [e["tool"] for e in ledger.all()] == ["lookup_property", "get_aerial", "vision_inspect"]
    assert "lookup_property failed" in t["lookup_property"].invoke({"query": "nowhere"})


def test_interior_inspect_skips_or_reports_measured_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = Ledger()
    assert tools("mock", ledger)["interior_inspect"].invoke({"property_id": 990001})["available"] is False
    monkeypatch.setattr(matterport, "model_id_for", lambda pid: None)
    skipped = tools("real", ledger)["interior_inspect"].invoke({"property_id": 7})
    assert skipped["available"] is False and "no Matterport scan" in skipped["reason"]
    monkeypatch.setattr(matterport, "model_id_for", lambda pid: "abc")
    monkeypatch.setattr(matterport, "get_model", lambda mid: matterport.MatterportModel(id=mid, metadata=matterport.summarize(MODEL)))
    scan = tools("real", ledger)["interior_inspect"].invoke({"property_id": 7})
    assert scan["available"] and scan["summary"]["room_count"] == 2
    assert scan["not_observed"] == ["lighting type", "window characteristics", "visible HVAC equipment"]


def test_carport_formula_and_not_estimated_upgrades() -> None:
    building = {"id": 1, "market_id": 20}
    rooftop = {"property_id": 1, "capacity_kw": 50.0, "annual_energy_kwh": 77500, "payback_years": 6.0,
               "usable_roof_area_sq_ft": 4167, "assumptions": {"estimated_cost_usd": 100000, "estimated_annual_savings_usd": 17050}}
    result = upgrades.screen(building, {"parking_space_count": 100, "observations": {"rooftop_obstructions": "few"}}, rooftop)
    by_name = {u["name"]: u for u in result["upgrades"]}
    carport = by_name["Solar carport"]
    kw = 100 * 0.6 * 162 * 15 / 1000                       # 145.8 kW
    kwh = kw * (77500 / 50.0)                              # same building's kWh per kW
    cost = kw * 1000 * 3.0
    savings = kwh * solar.PRICE_PER_KWH[20]
    assert carport["capacity_kw"] == round(kw, 1) and carport["annual_kwh"] == round(kwh)
    assert carport["cost_usd"] == round(cost) and carport["simple_payback_years"] == round(cost / savings, 1)
    assert carport["assumptions"]["kwh_per_kw"]["source"].startswith("same building")
    for name in ("LED lighting", "Window film", "HVAC upgrade"):
        assert by_name[name]["status"] == "not_estimated" and by_name[name]["reason"]
    assert "'few' rooftop obstructions" in by_name["HVAC upgrade"]["reason"]
    assert [u["name"] for u in sorted([u for u in result["upgrades"] if "rank" in u], key=lambda u: u["rank"])] == \
        ["Rooftop solar", "Solar carport"]  # ranked by simple payback
    assert "not a complete financial feasibility" in result["caveats"][0]


def test_no_parking_and_no_roof_mean_not_estimated() -> None:
    result = upgrades.screen({"id": 2, "market_id": 14}, {}, {"property_id": 2, "skipped": True, "reason": "no measured roof area"})
    by_name = {u["name"]: u for u in result["upgrades"]}
    assert by_name["Rooftop solar"]["status"] == by_name["Solar carport"]["status"] == "not_estimated"
    assert result["estimated_count"] == 0


def test_screen_upgrades_tool_records_ledger_entry() -> None:
    ledger = Ledger()
    t = tools("mock", ledger, intent="retrofit")
    b = t["sample_buildings"].invoke({"market_id": 20, "property_type": 11})["buildings"][0]
    t["vision_inspect"].invoke({"property_id": b["id"]})
    t["solar_payback"].invoke({"property_id": b["id"]})
    out = t["screen_upgrades"].invoke({"property_id": b["id"]})
    assert out["source"] == "mock" and out["estimated_count"] >= 1
    assert ledger.get(out["evidence_id"])["tool"] == "screen_upgrades"


def test_office_vacancy_sampling_explains_why_it_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 1: CoStar has no office vacancy field, so the sample says so instead of failing silently."""
    offices = [Building(id=i, market_id=14, property_type=5, latitude=29.7, longitude=-95.4, year_built=1990,
                        building_class=2, total_num_units=None, vacancy_percent=None) for i in range(1, 8)]
    monkeypatch.setattr(costar_data, "find_buildings", lambda market_id, property_type, limit=20: offices)
    q = "Do office buildings with more parking spaces have higher vacancy rates in Houston?"
    ledger = Ledger()
    out = tools("real", ledger, question=q)["sample_buildings"].invoke({"market_id": 14, "property_type": 5})
    assert out["count"] == 0 and "total_num_units" not in ledger.get("E1")["args"]["required_fields"]
    assert out["missing_by_field"] == {"vacancy_percent": 7}
    assert "vacancy_percent is missing for all 7" in out["unavailable_reason"]
    assert "only for multifamily" in out["unavailable_reason"]


# ---- planner: property flow and DeepAgents plan events ---------------------------------

class PropertyAgent:
    """Fake orchestrator for the property flow, including a write_todos plan."""

    def __init__(self, t: list[Any]) -> None:
        self.tools = {x.name: x for x in t}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        printer = config["callbacks"][0]
        printer.on_tool_start({"name": "write_todos"}, "", inputs={"todos": [
            {"content": "Look up the property", "status": "in_progress"}, {"content": "Inspect it", "status": "pending"}]})
        printer.on_tool_start({"name": "write_file"}, "", inputs={"file_path": "/findings.md", "content": "E1 lookup"})
        found = self.tools["lookup_property"].invoke({"query": "990005"})
        b = found["building"]
        self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        self.tools["interior_inspect"].invoke({"property_id": b["id"]})
        self.tools["solar_payback"].invoke({"property_id": b["id"]})
        rank = self.tools["rank_buildings"].invoke({})
        up = self.tools["screen_upgrades"].invoke({"property_id": b["id"]})
        self.tools["draft_email"].invoke({"property_id": b["id"]})
        r, e = rank["ranked"][0], rank["evidence_id"]
        brief = (f"## Question\nproperty 990005\n\n## Property\n{b['name']} [{found['evidence_id']}] (demo data).\n\n## Recommended upgrades\n"
                 f"Rooftop solar payback {r['payback_years']} [{e}] (demo data); {up['estimated_count']} [{up['evidence_id']}] "
                 "upgrades estimated.\n\n## Next step\nThe owner email draft is ready for review and was not sent.\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_property_flow_end_to_end_with_plan_events() -> None:
    events: list[dict] = []
    run = planner.run_question("Evaluate property 990005", mode="mock", agent_factory=PropertyAgent, on_event=events.append)
    assert run.complete and run.verified, run.problems
    s = run.study
    assert [b.id for b in s.buildings] == [990005] and s.stats is None
    assert s.solar_results and s.outreach_drafts and s.measurements
    kinds = [e["event_type"] for e in events]
    assert kinds[0] == "plan" and "note" in kinds and kinds[-1] == "citation_check"
    assert events[0]["payload"]["todos"][0] == {"content": "Look up the property", "status": "in_progress"}
    saved = agent_api.events_from_run(run)
    types = {e.event_type for e in saved}
    assert {"plan", "note", "property_details", "interior", "upgrade_screening"} <= types
    upgrade = next(e for e in saved if e.event_type == "upgrade_screening")
    assert {u["name"] for u in upgrade.payload["upgrades"]} == {"Rooftop solar", "Solar carport", "LED lighting",
                                                                "Window film", "HVAC upgrade"}


def test_property_flow_requires_upgrade_screening() -> None:
    ledger = Ledger()
    ledger.add("lookup_property", {}, {"building": {"id": 1}}, "mock")
    ledger.add("vision_inspect", {}, {"id": "m", "property_id": 1}, "mock")
    assert planner.missing_steps(ledger, "property") == ["screen_upgrades"]


def test_api_routes_property_questions_to_the_property_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from backend import main

    seen: list[str] = []

    def capture(question, mode=None, n=40, intent="auto", **kw):
        seen.append(intent)
        raise RuntimeError("stop")

    monkeypatch.setattr(planner, "run_question", capture)
    client = TestClient(main.app)
    client.post("/api/ask-agent", json={"question": "Evaluate property 23453"})
    client.post("/api/ask-agent", json={"question": "Which LA buildings suit rooftop solar?", "mode": "retrofit"})
    assert seen == ["property", "retrofit"]


# ---- retrofit corrections and calibration ----------------------------------------------

def solar_result(pid: int, roof: float, kwh_per_kw: float = 1550.0) -> SolarResult:
    usable = roof * 0.6
    kw = usable * 12 / 1000
    cost, savings = kw * 2000, kw * kwh_per_kw * 0.22
    return SolarResult(id=f"s{pid}", property_id=pid, usable_roof_area_sq_ft=round(usable), capacity_kw=round(kw, 1),
                       annual_energy_kwh=round(kw * kwh_per_kw), payback_years=round(cost / savings, 1),
                       assumptions={"usable_fraction_of_roof": 0.6, "watts_per_usable_sq_ft": 12.0,
                                    "installed_cost_usd_per_watt": 2.0, "electricity_price_usd_per_kwh": 0.22,
                                    "energy_source": "NREL PVWatts v8", "estimated_cost_usd": round(cost),
                                    "estimated_annual_savings_usd": round(savings), "measured_roof_area_sq_ft": roof})


def retrofit_run() -> StudyRun:
    """A real-source retrofit study where building 2 has a lower yield, so roof size reorders payback."""
    buildings = [Building(id=i, name=f"LA Building {i}", market_id=20, latitude=34.0, longitude=-118.2) for i in (1, 2, 3)]
    measurements = [Measurement(id=f"m-{i}", property_id=i, roof_area_sq_ft=r, parking_space_count=40)
                    for i, r in ((1, 10000.0), (2, 8000.0), (3, 6000.0))]
    estimates = [solar_result(1, 10000, 1500), solar_result(2, 8000, 1600), solar_result(3, 6000, 1550)]
    ranked = solar.rank_buildings(estimates)
    ledger = Ledger()
    ledger.add("sample_buildings", {}, {"buildings": [b.model_dump(mode="json") for b in buildings], "count": 3}, "real")
    for m in measurements:
        ledger.add("vision_inspect", {"property_id": m.property_id}, m.model_dump(mode="json"), "real")
    for e in estimates:
        ledger.add("solar_payback", {"property_id": e.property_id}, e.model_dump(mode="json"), "real")
    rank = ledger.add("rank_buildings", {}, {"ranked": [r.model_dump(mode="json") for r in ranked], "count": 3,
                                            "estimated": 3, "skipped_no_roof": 0}, "real")
    top = ranked[0]
    ledger.add("screen_upgrades", {"property_id": top.property_id},
               upgrades.screen(buildings[top.property_id - 1].model_dump(mode="json"),
                               measurements[top.property_id - 1].model_dump(mode="json"), top.model_dump(mode="json")), "real")
    draft = OutreachDraft(id=f"outreach-{top.property_id}", property_id=top.property_id, subject="Solar", body_markdown="x",
                          caveats=["Draft for human review; CityLab never sends email."])
    e = rank["id"]
    rows = "\n".join(f"| {r.rank} | {r.property_id} [{e}] | {r.usable_roof_area_sq_ft} [{e}] | {r.capacity_kw} [{e}] | "
                     f"{r.annual_energy_kwh} [{e}] | {r.payback_years} [{e}] |" for r in ranked)
    brief = ("## Question\nWhich LA buildings suit rooftop solar?\n\n## Answer\nThe top candidate is clear.\n\n## Ranking\n"
             "| Rank | Property id | Usable roof area (sq ft) | Capacity (kW) | Annual energy (kWh) | Payback (years) |\n"
             f"|---|---|---|---|---|---|\n{rows}\n\n## Caveats\n- Payback ignores incentives.\n")
    assert verify_brief(brief, ledger) == []
    study = StudyResult(id="study-retro01", question="Which LA buildings suit rooftop solar?", buildings=buildings,
                        measurements=measurements, stats=None, brief_markdown=brief, caveats=[],
                        solar_results=ranked, outreach_drafts=[draft])
    return StudyRun(study=study, ledger=ledger)


def test_rescale_solar_is_linear_and_documented() -> None:
    old = solar_result(1, 10000, 1500)
    new = rescale_solar(old, 5000.0)
    assert new.capacity_kw == pytest.approx(old.capacity_kw / 2, abs=0.1)
    assert new.annual_energy_kwh == pytest.approx(old.annual_energy_kwh / 2, abs=1)
    assert new.payback_years == old.payback_years  # same $/W and yield -> same simple payback
    assert "no new PVWatts call" in new.assumptions["energy_source"]
    assert rescale_solar(old, None).capacity_kw is None


def test_retrofit_roof_correction_recomputes_solar_rank_upgrades_and_draft() -> None:
    run = retrofit_run()
    top = run.study.solar_results[0]
    out = apply_correction(run, f"m-{top.property_id}", MeasurementCorrection(roof_area_sq_ft=200.0, note="tiny roof"))
    s = out.run.study
    new_top = s.solar_results[0]
    corrected = next(r for r in s.solar_results if r.property_id == top.property_id)
    assert corrected.capacity_kw < top.capacity_kw
    assert out.regression_entry["tool"] == "rank_buildings" and s.stats is None
    tools_after = [e["tool"] for e in out.run.ledger.all()[len(run.ledger.all()):]]
    assert tools_after == ["correct_measurement", "solar_payback", "rank_buildings", "screen_upgrades", "draft_email"]
    assert any("Redrafted from the fixed template" in c for c in s.outreach_drafts[0].caveats)
    assert verify_brief(s.brief_markdown, out.run.ledger) == []
    assert CORRECTION_HEADING in s.brief_markdown and f"[{out.regression_entry['id']}]" in s.brief_markdown.split("## Ranking")[1]
    labels = [c["label"] for c in out.changes]
    assert labels[0].startswith("roof_area_sq_ft") and any(l.startswith("simple payback") for l in labels)
    assert run.study.solar_results[0].capacity_kw == top.capacity_kw  # original untouched
    assert new_top.property_id is not None


def test_retrofit_correction_through_the_api_and_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from backend import main

    run = retrofit_run()
    save_study_run(run)
    client = TestClient(main.app)
    mid = f"m-{run.study.solar_results[0].property_id}"
    body = client.post("/api/correct", json={"study_id": run.study.id, "measurement_id": mid,
                                              "corrections": {"roof_area_sq_ft": 9000}}).json()
    assert body["study"]["id"] == "study-retro01-c1"
    kinds = [e["event_type"] for e in body["events"]]
    assert kinds == ["correction", "tool_call", "upgrade_screening", "citation_check"]
    assert body["events"][0]["payload"]["recomputed"] == "rank_buildings" and body["events"][-1]["message"] == "PASS"
    assert body["events"][0]["payload"]["calibration"]["roof_area_sq_ft"]["corrections"] == 1
    for i, area in enumerate((5000, 4000), start=2):
        client.post("/api/correct", json={"study_id": f"study-retro01-c{i - 1}", "measurement_id": "m-2",
                                          "corrections": {"roof_area_sq_ft": area}})
    cal = client.get("/api/calibration").json()["fields"]["roof_area_sq_ft"]
    assert cal["corrections"] == 3 and cal["reliability"] in ("ok", "low") and cal["confidence_multiplier"] is not None
    assert "not retrained" in client.get("/api/calibration").json()["method"]


def test_calibration_needs_enough_corrections() -> None:
    ledger = Ledger()
    ledger.add("correct_measurement", {"before": {"parking_space_count": 10}, "after": {"parking_space_count": 20}}, {}, "real")
    save_study_run(StudyRun(study=StudyResult(id="study-cal1", question="q", buildings=[], measurements=[], stats=None,
                                              brief_markdown="", caveats=[]), ledger=ledger))
    field = calibration_summary()["fields"]["parking_space_count"]
    assert field == {"corrections": 1, "median_relative_error": 1.0, "mean_signed_change": 1.0,
                     "reliability": "insufficient corrections", "confidence_multiplier": None}


class MockRetrofitAgent:
    """Fake orchestrator: demo retrofit study with ranking and drafts."""

    def __init__(self, t: list[Any]) -> None:
        self.tools = {x.name: x for x in t}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        s = self.tools["sample_buildings"].invoke({"market_id": 20, "property_type": 11})
        for b in s["buildings"]:
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
            self.tools["solar_payback"].invoke({"property_id": b["id"]})
        rank = self.tools["rank_buildings"].invoke({})
        self.tools["draft_email"].invoke({"property_id": rank["ranked"][0]["property_id"]})
        e = rank["evidence_id"]
        brief = f"## Question\nq\n\n## Ranking\nTop is {rank['ranked'][0]['property_id']} [{e}] (demo data).\n"
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_mock_retrofit_correction_says_demo_numbers_are_fixed() -> None:
    run = planner.run_question("Which 10 Los Angeles apartment buildings are the best rooftop-solar candidates?",
                               mode="mock", agent_factory=MockRetrofitAgent)
    assert run.complete, run.problems
    m = run.study.measurements[0]
    out = apply_correction(StudyRun(study=run.study, ledger=run.ledger), m.id, MeasurementCorrection(roof_area_sq_ft=100.0))
    assert "Demo studies use fixed illustrative solar numbers" in out.conclusion
    assert out.run.study.is_mock and out.regression_entry["source"] == "mock"
    assert verify_brief(out.run.study.brief_markdown, out.run.ledger) == []


def test_build_agent_gives_the_orchestrator_a_real_todo_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """deepagents 0.7.38 has no default to-do tool; build_agent adds LangChain's TodoListMiddleware."""
    from langchain.agents.middleware import TodoListMiddleware

    creator = Mock(return_value="agent")
    monkeypatch.setattr(planner, "create_deep_agent", creator)
    planner.build_agent(registry.get_tools("mock", Ledger(), intent="property"), object(), "property")
    middleware = creator.call_args.kwargs["middleware"]
    assert any(isinstance(m, TodoListMiddleware) for m in middleware)
    assert [t.name for m in middleware for t in m.tools] == ["write_todos"]
    names = [s["name"] for s in creator.call_args.kwargs["subagents"]]
    assert names == ["lookup", "inspector", "interior", "solar", "outreach", "writer"]
