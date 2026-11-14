"""Planner tests: fake agents drive real registry tools offline; mocks match contracts."""

import inspect
import json
import runpy
import socket
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, TypeAdapter

from backend.agent import planner
from backend.schemas import SolarResult, StudyResult
from backend.tools import mock

QUESTION = "In Houston, do apartment buildings with more parking spaces have higher asking rent?"

CANONICAL = (
    ("costar_data", "load_properties"), ("costar_data", "get_property"), ("costar_data", "find_buildings"),
    ("vexcel", "get_aerial"), ("vexcel", "get_property_info"), ("vision", "measure_building"),
    ("stats", "run_study"), ("solar", "estimate_solar"), ("solar", "rank_buildings"),
    ("outreach", "draft_outreach"), ("matterport", "get_model"),
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tracing and fail every attempted network connection in these tests."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def forbidden(*args: object, **kwargs: object) -> None:
        """Network access is never part of a fake-agent test."""
        raise AssertionError("Network access attempted in an offline planner test")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


class FakeAgent:
    """Calls the registry tools in sampler -> inspector -> statistician order, then writes."""

    def __init__(self, tools: list[Any], cite: bool = True) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.cite = cite
        self.calls = 0

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the workflow on the first call; on a fix request, return the same brief."""
        self.calls += 1
        if self.calls == 1:
            sample = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
            for b in sample["buildings"]:
                self.tools["vexcel_detect"].invoke({"lat": b["latitude"], "lon": b["longitude"]})
                self.tools["vision_inspect"].invoke({"property_id": b["id"]})
            stats = self.tools["run_regression"].invoke({"question": QUESTION})
            self.sample, self.stats = sample, stats
        s, st = self.sample, self.stats
        p = st["predictor"]
        lo, hi = st["confidence_intervals"][p]
        extra = "" if self.cite else " Rents rose 37 percent overall."
        brief = (
            f"## Question\n{QUESTION}\n\n## Answer\nNo clear association (demo data).\n\n"
            f"## Method\nWe sampled n = {s['count']} [{s['evidence_id']}] Houston apartments (demo data).\n\n"
            f"## Result\nThe coefficient is {st['coefficients'][p]} [{st['evidence_id']}] (95% CI "
            f"{lo} to {hi} [{st['evidence_id']}]), p = {st['p_values'][p]} [{st['evidence_id']}] (demo data).{extra}\n\n"
            "## Confidence and caveats\n1. Correlation is not causation.\n2. The sample is small.\n"
        )
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_run_question_with_fake_agent_returns_verified_study(capsys: pytest.CaptureFixture[str]) -> None:
    """A fully cited brief verifies and yields a schema-valid StudyResult."""
    agents: list[FakeAgent] = []
    run = planner.run_question(QUESTION, mode="mock", agent_factory=lambda t: agents.append(FakeAgent(t)) or agents[-1])
    assert run.verified and run.problems == []
    assert agents[0].calls == 1
    StudyResult.model_validate(run.study.model_dump(mode="json"))
    assert run.study.is_mock and len(run.study.buildings) == 12 and len(run.study.measurements) == 12
    assert run.study.stats is not None and run.study.stats.predictor == "parking_space_count"
    tools = [e["tool"] for e in run.ledger.all()]
    assert tools[0] == "sample_buildings" and tools[-1] == "run_regression"
    assert tools[1:4] == ["vexcel_detect", "get_aerial", "vision_inspect"]
    out = capsys.readouterr().out
    assert "TOOL E1 sample_buildings" in out and '"limit": 40' in out


def test_uncited_number_fails_after_one_fix_attempt() -> None:
    """An uncited number survives the single fix round, so a caveat is added."""
    agents: list[FakeAgent] = []
    run = planner.run_question(
        QUESTION, mode="mock", agent_factory=lambda t: agents.append(FakeAgent(t, cite=False)) or agents[-1])
    assert agents[0].calls == 2  # one writer fix attempt
    assert run.verified is False
    assert any("37" in p for p in run.problems)
    assert any(c.startswith("Unverified numbers") for c in run.study.caveats)
    assert "Unverified numbers" in run.study.brief_markdown


class SkipRegressionAgent:
    """Samples and inspects but never runs the statistician, even when told to."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record the last instruction and write a brief citing only the sample."""
        self.calls += 1
        self.prompts.append(str(inputs["messages"][-1].content))
        if self.calls == 1:
            self.sample = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
            b = self.sample["buildings"][0]
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        brief = (f"## Question\n{QUESTION}\n\n## Method\nWe sampled n = {self.sample['count']} "
                 f"[{self.sample['evidence_id']}] buildings (demo data).\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_missing_regression_triggers_one_guard_then_caveat(capsys: pytest.CaptureFixture[str]) -> None:
    """The guard asks once for the missing step; if still missing, the study is unverified."""
    agents: list[SkipRegressionAgent] = []
    run = planner.run_question(
        QUESTION, mode="mock", agent_factory=lambda t: agents.append(SkipRegressionAgent(t)) or agents[-1])
    agent = agents[0]
    assert agent.calls == 2  # initial run + exactly one corrective message
    assert "run_regression was not run" in agent.prompts[1] and "statistician" in agent.prompts[1]
    assert run.verified is False
    assert run.complete is False
    assert "Study incomplete: regression did not produce a result" in run.study.caveats
    assert "Study incomplete: regression did not produce a result" in run.study.brief_markdown
    assert planner.verdict_line(run) == "CITATION CHECK: FAIL (study incomplete)"
    assert run.study.stats is None
    assert "STEP GUARD: run_regression" in capsys.readouterr().out


def test_guard_passes_when_the_retry_runs_the_missing_step() -> None:
    """If the corrective run adds run_regression, no incomplete caveat is added."""
    class LateAgent(FakeAgent):
        def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
            if self.calls == 0:
                self.calls += 1
                self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
                return {"messages": [*inputs["messages"], AIMessage(content="## Question\nPending.")]}
            self.calls = 0  # let FakeAgent run its full workflow on the guard retry
            return super().invoke(inputs, config)

    run = planner.run_question(QUESTION, mode="mock", agent_factory=LateAgent)
    assert run.verified is True
    assert not any(c.startswith("Study incomplete") for c in run.study.caveats)


class NoResultRegressionAgent(SkipRegressionAgent):
    """Runs run_regression, but on a question the tool cannot use (no outcome), twice."""

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        state = super().invoke(inputs, config)
        self.no_result = self.tools["run_regression"].invoke({"question": "Is anything associated with parking?"})
        return state


def test_regression_without_a_result_is_incomplete_not_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A StatsResult with no coefficient (findings.error) must not count as a finished study."""
    from backend.schemas import StatsResult
    from backend.tools import mock as mock_tools

    monkeypatch.setattr(mock_tools, "run_study", lambda question, buildings, measurements=None: StatsResult(
        id="s", predictor="parking_space_count", sample_size=0, findings={"error": "not enough complete rows (n=0)"}))
    agents: list[NoResultRegressionAgent] = []
    run = planner.run_question(
        QUESTION, mode="mock", agent_factory=lambda t: agents.append(NoResultRegressionAgent(t)) or agents[-1])
    assert agents[0].calls == 2  # the guard fires once even though run_regression was called
    assert "did not produce a result" in agents[0].prompts[1]
    assert run.verified is False and run.complete is False
    assert run.problems[-1] == "Study incomplete: regression did not produce a result"
    assert run.study.stats is not None and run.study.stats.coefficients is None


def test_cli_fails_and_exits_nonzero_for_incomplete_study(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prints FAIL (study incomplete) and exits non-zero."""
    real_run = planner.run_question
    monkeypatch.setattr(planner, "run_question", lambda q, mode=None, n=40, intent="auto": real_run(
        q, mode=mode, n=n, agent_factory=SkipRegressionAgent, intent=intent))
    assert planner.main([QUESTION, "--mode", "mock"]) != 0
    assert "CITATION CHECK: FAIL (study incomplete)" in capsys.readouterr().out


def test_complete_study_verdict_still_passes() -> None:
    """A complete, cited study prints PASS."""
    run = planner.run_question(QUESTION, mode="mock", agent_factory=FakeAgent)
    assert run.complete and run.verified
    assert planner.verdict_line(run) == "CITATION CHECK: PASS"


def test_build_agent_wires_four_subagents_with_their_own_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_deep_agent receives the orchestrator prompt and per-subagent tools."""
    from backend.agent.evidence import Ledger
    from backend.agent.registry import get_tools

    creator = Mock(return_value="agent")
    monkeypatch.setattr(planner, "create_deep_agent", creator)
    model = object()
    assert planner.build_agent(get_tools("mock", Ledger()), model) == "agent"
    kwargs = creator.call_args.kwargs
    assert kwargs["model"] is model and kwargs["name"] == "CityLab"
    assert "Never invent numbers" in kwargs["system_prompt"]
    subagents = {s["name"]: [t.name for t in s["tools"]] for s in kwargs["subagents"]}
    assert subagents == {
        "sampler": ["sample_buildings"], "inspector": ["vexcel_detect", "vision_inspect"],
        "statistician": ["run_regression"], "writer": ["read_evidence", "list_evidence"],
    }
    assert {t.name for t in kwargs["tools"]} == {"list_evidence", "read_evidence"}


def test_brief_prefers_writer_subagent_output() -> None:
    """The writer's task result is the brief even if the orchestrator paraphrases it."""
    call = {"name": "task", "args": {"subagent_type": "writer", "description": "write"}, "id": "w1"}
    from langchain_core.messages import ToolMessage
    messages = [HumanMessage(content="q"), AIMessage(content="", tool_calls=[call]),
                ToolMessage(content="Here are 2 fixes.\n\n## Question\nwriter brief", tool_call_id="w1"),
                AIMessage(content="paraphrase")]
    assert planner._brief_from(messages) == "## Question\nwriter brief"


def test_handoff_printer_prints_subagent_and_redacts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Task tool calls print as hand-offs; other tools are printed by the registry."""
    monkeypatch.setenv("VEXCEL_API_KEY", "private-test-value")
    printer = planner._HandOffPrinter()
    printer.on_tool_start({"name": "task"}, "", inputs={"subagent_type": "sampler",
                                                       "description": "sample private-test-value"})
    printer.on_tool_start({"name": "vexcel_detect"}, "", inputs={"lat": 1})
    out = capsys.readouterr().out
    assert out.startswith("HAND-OFF -> sampler: sample [REDACTED]")
    assert "vexcel_detect" not in out and "private-test-value" not in out


def test_cli_prints_check_and_tools_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any,
) -> None:
    """The CLI prints the brief, the citation verdict, the tools table and writes JSON."""
    real_run = planner.run_question

    def run(question: str, mode: str | None = None, n: int = 40, intent: str = "auto") -> planner.PlannerRun:
        return real_run(question, mode=mode, n=n, agent_factory=FakeAgent, intent=intent)

    monkeypatch.setattr(planner, "run_question", run)
    path = tmp_path / "out.json"
    assert planner.main([QUESTION, "--mode", "mock", "--json", str(path)]) == 0
    out = capsys.readouterr().out
    assert "CITATION CHECK: PASS" in out and "TOOLS USED:" in out
    assert any(line.split()[:3] == ["vision_inspect", "mock", "12"] for line in out.splitlines())
    data = json.loads(path.read_text())
    assert data["verified"] is True and data["ledger"][0]["id"] == "E1"
    StudyResult.model_validate(data["study"])


def test_cli_hides_exception_details(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Provider exception text never reaches the console."""
    monkeypatch.setattr(planner, "run_question", Mock(side_effect=RuntimeError("private-sdk-detail")))
    with pytest.raises(SystemExit) as error:
        planner.main(["Describe the sample"])
    assert error.value.code == 1
    assert "private-sdk-detail" not in capsys.readouterr().err


def test_empty_question_does_not_create_an_agent() -> None:
    """A blank question fails before any tools or model are built."""
    factory = Mock()
    with pytest.raises(ValueError, match="nonempty"):
        planner.run_question("  ", agent_factory=factory)
    factory.assert_not_called()


# ---- mock tool contracts (kept from the earlier planner tests) -------------------------

def test_every_mock_matches_a_contract_and_returns_schema_objects() -> None:
    """Cover all eleven mock functions, including optional Matterport, without keys."""
    building = mock.find_buildings(14, 11, 1)[0]
    image = mock.get_aerial(building.latitude, building.longitude, building.id)
    measurement = mock.measure_building(image.image_path, building.id)
    solar = mock.estimate_solar(building, measurement)
    arguments = {
        "load_properties": {}, "get_property": {"id": building.id},
        "find_buildings": {"market_id": 14, "property_type": 11, "limit": 2},
        "get_aerial": {"lat": building.latitude, "lon": building.longitude, "property_id": building.id},
        "get_property_info": {"lat": building.latitude, "lon": building.longitude},
        "measure_building": {"image_path": image.image_path, "property_id": building.id},
        "run_study": {"question": "green cover and rent", "buildings": [building], "measurements": [measurement]},
        "estimate_solar": {"building": building, "vision_result": measurement},
        "rank_buildings": {"estimates": [solar]},
        "draft_outreach": {"building": building, "solar_estimate": solar},
        "get_model": {"model_id": "mock-model"},
    }
    for module_name, function_name in CANONICAL:
        real_module = __import__(f"backend.tools.{module_name}", fromlist=[function_name])
        real = getattr(real_module, function_name)
        function = getattr(mock, function_name)
        assert inspect.signature(function, eval_str=True) == inspect.signature(real, eval_str=True), function_name
        result = function(**arguments[function_name])
        objects = result if isinstance(result, list) else [result]
        assert all(isinstance(item, BaseModel) for item in objects), function_name
        adapter = TypeAdapter(inspect.signature(function).return_annotation)
        assert adapter.validate_json(adapter.dump_json(result)) == result
    assert mock.get_property(-1) is None


def test_mock_regression_follows_the_question_predictor() -> None:
    """Parking, vacancy and default (green cover) questions get matching fixed predictors."""
    buildings = mock.find_buildings(14, 11, 12)
    assert mock.run_study("parking and rent", buildings).predictor == "parking_space_count"
    assert mock.run_study("vacancy and paving", buildings).predictor == "paved_surface_percent"
    assert mock.run_study("green cover and rent", buildings).predictor == "green_cover_percent"


def test_mock_cohorts_are_deterministic_and_do_not_share_mutable_models() -> None:
    """Separate city ids/locations and fresh objects prevent fake-data contamination."""
    houston = mock.find_buildings(14, 11, 50)
    la = mock.find_buildings(20, 11, 50)
    assert len(houston) == len(la) == 12
    assert {b.id for b in houston}.isdisjoint(b.id for b in la)
    assert all(b.market_id == 20 and 33 < b.latitude < 35 and -119 < b.longitude < -117 for b in la)
    original_name = houston[0].name
    houston[0].name = "changed by caller"
    assert mock.find_buildings(14, 11, 1)[0].name == original_name
    assert mock.find_buildings(14, 5, 10) == []


@pytest.mark.parametrize("setting, expected", [(None, True), ("true", True), ("false", False), (" TRUE ", True)])
def test_mock_mode_setting(setting: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """USE_MOCK_TOOLS (used by the web study) still parses; mocks are its default."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    if setting is None:
        monkeypatch.delenv("USE_MOCK_TOOLS", raising=False)
    else:
        monkeypatch.setenv("USE_MOCK_TOOLS", setting)
    from backend import config
    assert runpy.run_path(config.__file__)["USE_MOCK_TOOLS"] is expected


def test_solar_rankings_use_payback_and_preserve_inputs() -> None:
    """Missing payback ranks last, with one-based ranks and no email being sent."""
    inputs = [SolarResult(id="unknown", property_id=1),
              SolarResult(id="slow", property_id=2, payback_years=9),
              SolarResult(id="fast", property_id=3, payback_years=6)]
    ranked = mock.rank_buildings(inputs)
    assert [result.id for result in ranked] == ["fast", "slow", "unknown"]
    assert [result.rank for result in ranked] == [1, 2, 3]
    assert all(result.rank is None for result in inputs)


# ---- retrofit mode -------------------------------------------------------------------

RETROFIT_Q = "Which 10 Los Angeles apartment buildings are the best rooftop-solar candidates?"


class RetrofitAgent:
    """Fake retrofit orchestrator; skip_drafts leaves out the outreach step."""

    def __init__(self, tools: list[Any], skip_drafts: bool = False) -> None:
        self.tools = {t.name: t for t in tools}
        self.skip_drafts = skip_drafts
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls += 1
        self.prompts.append(str(inputs["messages"][-1].content))
        if self.calls == 1:
            s = self.tools["sample_buildings"].invoke({"market_id": 20, "property_type": 11})
            for b in s["buildings"]:
                self.tools["vision_inspect"].invoke({"property_id": b["id"]})
                self.tools["solar_payback"].invoke({"property_id": b["id"]})
            self.rank = self.tools["rank_buildings"].invoke({})
            if not self.skip_drafts:
                for r in self.rank["ranked"][:10]:
                    self.tools["draft_email"].invoke({"property_id": r["property_id"]})
        e = self.rank["evidence_id"]
        rows = "\n".join(
            f"| {r['rank']} | {r['property_id']} [{e}] | {r['usable_roof_area_sq_ft']} [{e}] | "
            f"{r['capacity_kw']} [{e}] | {r['annual_energy_kwh']} [{e}] | {r['payback_years']} [{e}] |"
            for r in self.rank["ranked"][:10])
        brief = (f"## Question\n{RETROFIT_Q}\n\n## Answer\nProperty {self.rank['ranked'][0]['property_id']} [{e}] "
                 "is the top candidate (demo data).\n\n## Ranking\n"
                 "| Rank | Property id | Usable roof area (sq ft) | Capacity (kW) | Annual energy (kWh) | Payback (years) |\n"
                 f"|---|---|---|---|---|---|\n{rows}\n\n## Caveats\n1. Payback ignores incentives and financing.\n"
                 f"2. Roof area is image-measured; {self.rank['skipped_no_roof']} [{e}] skipped.\n\n"
                 "## Next step\nThe first email draft is available for review.\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def test_retrofit_run_builds_ranked_study_with_drafts() -> None:
    """Retrofit: stats None, ranked solar_results, ten drafts, verified table citations."""
    run = planner.run_question(RETROFIT_Q, mode="mock", agent_factory=RetrofitAgent)
    assert run.verified and run.complete, run.problems
    study = run.study
    StudyResult.model_validate(study.model_dump(mode="json"))
    assert study.stats is None and len(study.solar_results) == 12 and len(study.outreach_drafts) == 10
    assert [r.rank for r in study.solar_results] == list(range(1, 13))
    assert {d.property_id for d in study.outreach_drafts} == {r.property_id for r in study.solar_results[:10]}
    assert len(study.measurements) == 12 and len(study.buildings) == 12
    assert "run_regression" not in [e["tool"] for e in run.ledger.all()]
    assert planner.verdict_line(run) == "CITATION CHECK: PASS"


def test_retrofit_without_drafts_is_incomplete() -> None:
    """No outreach draft: guard fires once, then caveat and FAIL."""
    agents: list[RetrofitAgent] = []
    run = planner.run_question(RETROFIT_Q, mode="mock", intent="retrofit",
                               agent_factory=lambda t: agents.append(RetrofitAgent(t, skip_drafts=True)) or agents[-1])
    assert agents[0].calls == 2 and "outreach" in agents[0].prompts[1]
    assert run.verified is False and run.complete is False
    assert "Study incomplete: draft_email was not run." in run.study.caveats
    assert planner.verdict_line(run) == "CITATION CHECK: FAIL (study incomplete)"


def test_retrofit_missing_ranking_is_incomplete() -> None:
    """A ledger without a successful rank_buildings is incomplete for retrofit."""
    from backend.agent.evidence import Ledger

    ledger = Ledger()
    ledger.add("sample_buildings", {}, {"buildings": [], "count": 0}, "mock")
    ledger.add("vision_inspect", {}, {"id": "m", "property_id": 1}, "mock")
    ledger.add("rank_buildings", {}, {"error": "no solar estimates"}, "mock")
    assert planner.missing_steps(ledger, "retrofit") == ["rank_buildings", "draft_email"]
    assert planner.missing_steps(ledger, "research") == ["run_regression"]


def test_intent_auto_and_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto detects retrofit from the question; explicit intent overrides it; CLI passes --intent."""
    seen: list[str] = []

    def factory(intent: str):
        seen.append(intent)
        return FakeAgent if intent == "research" else RetrofitAgent

    monkeypatch.setattr(planner, "_default_factory", factory)
    assert planner.run_question(RETROFIT_Q, mode="mock").study.solar_results
    assert planner.run_question(QUESTION, mode="mock").study.stats is not None
    planner.run_question(QUESTION, mode="mock", intent="research")
    assert seen == ["retrofit", "research", "research"]
    run = Mock(return_value=planner.run_question(RETROFIT_Q, mode="mock"))
    monkeypatch.setattr(planner, "run_question", run)
    planner.main([RETROFIT_Q, "--mode", "mock", "--intent", "retrofit"])
    assert run.call_args.kwargs["intent"] == "retrofit"


def test_build_agent_retrofit_subagents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrofit wires sampler, inspector, solar, outreach and the retrofit writer."""
    from backend.agent import prompts
    from backend.agent.evidence import Ledger
    from backend.agent.registry import get_tools

    creator = Mock(return_value="agent")
    monkeypatch.setattr(planner, "create_deep_agent", creator)
    planner.build_agent(get_tools("mock", Ledger(), intent="retrofit"), object(), "retrofit")
    subs = {s["name"]: s for s in creator.call_args.kwargs["subagents"]}
    assert {n: [t.name for t in s["tools"]] for n, s in subs.items()} == {
        "sampler": ["sample_buildings"], "inspector": ["vexcel_detect", "vision_inspect"],
        "solar": ["solar_payback", "rank_buildings", "screen_upgrades"], "outreach": ["draft_email"],
        "writer": ["read_evidence", "list_evidence"]}
    assert subs["writer"]["system_prompt"] == prompts.RETROFIT_WRITER_PROMPT
    assert "statistician" not in subs
