"""Correction loop (POST /api/correct): offline, no LLM, no network."""

import json
import socket
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import config, main
from backend.agent import planner
from backend.agent.citations import verify_brief
from backend.agent.correction import (CORRECTION_HEADING, CorrectionNotFound, InvalidCorrection,
                                      _num as num, apply_correction)
from backend.agent.evidence import Ledger
from backend.agent.study_store import StudyRun, load_study_run, save_study_run, study_path
from backend.schemas import Building, CorrectResponse, Measurement, MeasurementCorrection, StudyResult
from backend.tools import stats

Q = "In Houston, do apartment buildings with more parking spaces have higher asking rent?"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any socket use (LLM, Vexcel, PVWatts) fails the test."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted during a correction")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def real_run(n: int = 30, seed: int = 3) -> StudyRun:
    """A saved-style research study with a real HC3 regression of rent on parking."""
    rng = np.random.default_rng(seed)
    buildings, measurements = [], []
    for i in range(n):
        parking = int(rng.integers(20, 300))
        size = float(rng.uniform(30_000, 300_000))
        year, cls = int(rng.integers(1960, 2020)), int(rng.integers(1, 4))
        rent = 900 + 2.5 * parking + 0.001 * size + 3 * (year - 1960) + 120 * (3 - cls) + float(rng.normal(0, 60))
        buildings.append(Building(id=1000 + i, name=f"Test Community {i}", building_size_sq_ft=size,
                                  year_built=year, building_class=cls, total_num_units=int(rng.integers(40, 500)),
                                  max_rent=round(rent, 2), latitude=29.7, longitude=-95.4))
        measurements.append(Measurement(id=f"m-{1000 + i}", property_id=1000 + i, parking_space_count=parking,
                                        green_cover_percent=float(rng.uniform(5, 40)), observations={"notes": "ok"}))
    ledger = Ledger()
    sample = ledger.add("sample_buildings", {"market_id": 14, "property_type": 11, "limit": n},
                        {"buildings": [b.model_dump(mode="json") for b in buildings], "count": n}, "real")
    for m in measurements:
        ledger.add("vision_inspect", {"image_path": "x.jpg", "property_id": m.property_id},
                   m.model_dump(mode="json"), "real")
    result = stats.run_study(Q, buildings, measurements)
    reg = ledger.add("run_regression", {"question": Q, "property_ids": [b.id for b in buildings],
                                        "measurement_count": n}, result.model_dump(mode="json"), "real")
    e, s = reg["id"], result
    p = s.predictor
    lo, hi = s.confidence_intervals[p]
    brief = (f"## Question\n{Q}\n\n## Answer\nMore parking goes with higher rent.\n\n"
             f"## Method\nWe sampled n = {n} [{sample['id']}] buildings.\n\n"
             f"## Result\nThe coefficient is {num(s.coefficients[p])} [{e}] (95% CI {num(lo)} [{e}] to {num(hi)} [{e}]), "
             f"p = {num(s.p_values[p])} [{e}], n = {s.sample_size} [{e}].\n\n"
             "## Confidence and caveats\n1. Correlation is not causation.\n2. Small sample.\n")
    assert verify_brief(brief, ledger) == []
    study = StudyResult(id="study-test0001", question=Q, buildings=buildings, measurements=measurements,
                        stats=result, brief_markdown=brief, caveats=list(result.caveats or []))
    return StudyRun(study=study, ledger=ledger)


def correction(**fields: Any) -> MeasurementCorrection:
    return MeasurementCorrection(**fields)


def test_parking_correction_changes_coefficient_and_what_changed() -> None:
    run = real_run()
    old = run.study.stats.coefficients["parking_space_count"]
    target = run.study.measurements[0]
    out = apply_correction(run, target.id, correction(parking_space_count=target.parking_space_count + 400))
    new_stats = out.run.study.stats
    assert new_stats.coefficients["parking_space_count"] != pytest.approx(old)
    assert (new_stats.outcome, new_stats.predictor) == (run.study.stats.outcome, run.study.stats.predictor)
    brief = out.run.study.brief_markdown
    old_e, new_e, cor_e = out.previous_regression_entry["id"], out.regression_entry["id"], out.correction_entry["id"]
    assert CORRECTION_HEADING in brief
    assert f"{target.parking_space_count} [{cor_e}] before, {target.parking_space_count + 400} [{cor_e}] after" in brief
    assert f"[{old_e}] before" in brief and f"[{new_e}] after" in brief
    result_section = brief.split("## Result")[1].split("## ")[0]
    assert f"[{new_e}]" in result_section and f"[{old_e}]" not in result_section
    assert out.conclusion.startswith("The conclusion")
    coef = next(c for c in out.changes if c["label"] == "parking_space_count coefficient")
    assert coef["before"] == pytest.approx(old) and coef["direction"] in ("up", "down")
    field = out.changes[0]
    assert field["label"] == "parking_space_count (Test Community 0)" and field["direction"] == "up"
    entry = out.regression_entry
    assert entry["tool"] == "run_regression" and entry["source"] == "real"
    assert entry["args"]["note"] == f"after correction of {target.id}"
    assert int(new_e[1:]) == int(cor_e[1:]) + 1 == len(run.ledger.all()) + 2


def test_final_brief_passes_verify_brief() -> None:
    run = real_run()
    out = apply_correction(run, run.study.measurements[3].id,
                           correction(parking_space_count=5, note="I counted 5 spaces in 2 rows."))
    assert verify_brief(out.run.study.brief_markdown, out.run.ledger) == []
    assert "Your note is saved with the correction" in out.run.study.brief_markdown
    assert any("I counted 5 spaces" in c for c in out.run.study.caveats)


def test_omitted_fields_unchanged_and_explicit_null_clears() -> None:
    run = real_run()
    m = run.study.measurements[1]
    only_parking = apply_correction(run, m.id, correction(parking_space_count=7)).corrected_measurement
    assert only_parking.green_cover_percent == m.green_cover_percent and only_parking.parking_space_count == 7
    cleared = apply_correction(run, m.id, correction(green_cover_percent=None)).corrected_measurement
    assert cleared.green_cover_percent is None and cleared.parking_space_count == m.parking_space_count
    assert cleared.is_corrected is True and cleared.observations["corrections"][-1]["fields"] == ["green_cover_percent"]


def test_clearing_the_predictor_drops_that_row() -> None:
    """An explicit null on the predictor makes it unknown, so the re-run analyzes one row fewer."""
    run = real_run()
    out = apply_correction(run, run.study.measurements[2].id, correction(parking_space_count=None))
    assert out.run.study.stats.sample_size == run.study.stats.sample_size - 1
    n_change = next(c for c in out.changes if c["label"] == "sample size")
    assert n_change["direction"] == "down"


def test_invalid_values_and_empty_corrections_are_rejected() -> None:
    run = real_run()
    mid = run.study.measurements[0].id
    with pytest.raises(InvalidCorrection):
        apply_correction(run, mid, MeasurementCorrection.model_construct(parking_space_count=-3, _fields_set={"parking_space_count"}))
    with pytest.raises(InvalidCorrection):
        apply_correction(run, mid, MeasurementCorrection.model_construct(green_cover_percent=140.0, _fields_set={"green_cover_percent"}))
    with pytest.raises(InvalidCorrection):
        apply_correction(run, mid, correction(note="only a note"))
    with pytest.raises(CorrectionNotFound):
        apply_correction(run, "m-nope", correction(parking_space_count=1))


def test_new_id_and_original_unchanged_and_second_correction() -> None:
    run = real_run()
    save_study_run(run)
    before_text = study_path(run.study.id).read_text()
    first = apply_correction(load_study_run(run.study.id), run.study.measurements[0].id, correction(parking_space_count=250))
    save_study_run(first.run)
    assert first.run.study.id == "study-test0001-c1" and first.run.meta["corrected_from"] == "study-test0001"
    assert first.run.meta["original_id"] == "study-test0001"
    assert study_path(run.study.id).read_text() == before_text  # original file untouched
    assert len(run.ledger.all()) == len(load_study_run(run.study.id).ledger.all())
    second = apply_correction(load_study_run(first.run.study.id), run.study.measurements[5].id,
                              correction(parking_space_count=12))
    assert second.run.study.id == "study-test0001-c2" and second.run.meta["corrected_from"] == "study-test0001-c1"
    assert second.previous_regression_entry["id"] == first.regression_entry["id"]
    assert second.run.study.brief_markdown.count(CORRECTION_HEADING) == 1
    assert verify_brief(second.run.study.brief_markdown, second.run.ledger) == []
    corrected = {m.id: m for m in second.run.study.measurements}
    assert corrected[run.study.measurements[0].id].parking_space_count == 250  # earlier correction kept
    assert len(second.run.meta["corrections"]) == 2


# ---- API -------------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def post(client: TestClient, study_id: str, measurement_id: str, corrections: dict) -> Any:
    return client.post("/api/correct", json={"study_id": study_id, "measurement_id": measurement_id,
                                             "corrections": corrections})


def test_api_returns_corrected_study_events_and_measurement(client: TestClient) -> None:
    run = real_run()
    save_study_run(run)
    target = run.study.measurements[0]
    response = post(client, run.study.id, target.id, {"parking_space_count": 299, "note": "recounted"})
    assert response.status_code == 200, response.text
    body = CorrectResponse.model_validate(response.json())
    assert body.study.id == "study-test0001-c1" and body.corrected_measurement.parking_space_count == 299
    assert [e.event_type for e in body.events] == ["correction", "tool_call", "citation_check"]
    correction_event = body.events[0]
    assert correction_event.payload["changes"][0]["after"] == 299
    assert {"label", "before", "after", "direction"} <= set(correction_event.payload["changes"][0])
    assert body.events[1].payload["tool"] == "run_regression" and body.events[-1].message == "PASS"
    assert load_study_run(body.study.id).study.id == body.study.id  # saved under the new id


@pytest.mark.parametrize("corrections", [{"parking_space_count": -1}, {"green_cover_percent": 101},
                                         {"paved_surface_percent": -0.5}, {"roof_area_sq_ft": -10},
                                         {"unknown_field": 3}, {}])
def test_api_invalid_values_are_422(client: TestClient, corrections: dict) -> None:
    run = real_run()
    save_study_run(run)
    response = post(client, run.study.id, run.study.measurements[0].id, corrections)
    assert response.status_code == 422


def test_api_unknown_ids_are_404_without_internals(client: TestClient) -> None:
    run = real_run()
    save_study_run(run)
    for study_id, mid in (("study-missing", "m-1000"), ("../../.env", "m-1000"), (run.study.id, "m-nope")):
        response = post(client, study_id, mid, {"parking_space_count": 3})
        assert response.status_code == 404
        assert "Traceback" not in response.text and "data/cache" not in response.text


def test_api_works_with_cache_only(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    run = real_run()
    save_study_run(run)
    assert post(client, run.study.id, run.study.measurements[0].id, {"parking_space_count": 1}).status_code == 200


class MockStudyAgent:
    """Fake orchestrator: samples, inspects and regresses with mock tools, writes a cited brief."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = {t.name: t for t in tools}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        s = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
        for b in s["buildings"]:
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        st = self.tools["run_regression"].invoke({"question": Q})
        p, e = st["predictor"], st["evidence_id"]
        brief = (f"## Question\n{Q}\n\n## Answer\nNo clear association (demo data).\n\n## Result\n"
                 f"The coefficient is {st['coefficients'][p]} [{e}], n = {st['sample_size']} [{e}] (demo data).\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_end_to_end_mock_study_saved_by_ask_agent_then_corrected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/ask-agent saves the study; /api/correct works on it; demo numbers stay fixed."""
    real = planner.run_question
    monkeypatch.setattr(planner, "run_question", lambda q, mode=None, n=40, intent="auto", on_event=None:
                        real(q, mode=mode, n=n, intent=intent, on_event=on_event, agent_factory=MockStudyAgent))
    asked = client.post("/api/ask-agent", json={"question": Q, "use_mock_tools": True})
    assert asked.status_code == 200, asked.text
    study = asked.json()["study"]
    saved = load_study_run(study["id"])
    assert saved.study.is_mock and len(saved.ledger.all()) == len(asked.json()["events"]) - 1
    mid = study["measurements"][0]["id"]
    response = post(client, study["id"], mid, {"parking_space_count": 999})
    assert response.status_code == 200, response.text
    body = CorrectResponse.model_validate(response.json())
    assert body.study.is_mock and body.study.id == f"{study['id']}-c1"
    assert body.events[1].payload["source"] == "mock"
    conclusion = body.events[0].payload["conclusion"]
    assert "did not change" in conclusion and "Demo studies use fixed illustrative regression numbers" in conclusion
    assert body.events[-1].message == "PASS"
    assert json.loads(json.dumps(body.events[0].payload))  # JSON-safe


def test_digits_in_building_names_and_tiny_p_values_stay_citable() -> None:
    """A name like 'Park 45' is cited via the correction entry; p < 1e-6 is worded, not '0'."""
    from backend.agent.correction import _cite_p

    run = real_run()
    run.study.buildings[0].name = "Park 45 Apartments"
    out = apply_correction(run, run.study.measurements[0].id, correction(parking_space_count=40))
    assert "Park 45 Apartments" in out.run.study.brief_markdown
    assert verify_brief(out.run.study.brief_markdown, out.run.ledger) == []
    assert out.correction_entry["args"]["property_name"] == "Park 45 Apartments"
    assert _cite_p(3e-12, "E9") == "below one in a million [E9]" and _cite_p(0.031, "E9") == "0.031 [E9]"
    assert "p = 0 [" not in out.run.study.brief_markdown
