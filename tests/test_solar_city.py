"""The solar page's live run: DeepAgents orchestration, backend completion, SSE and the page script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.agent import solar_city as sc

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = list(sc.TOOLS)


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never send test traces to LangSmith."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class FakeCity:
    """Stands in for scripts.build_solar_demo.CityRun: same stages, fixed numbers, no I/O."""

    def __init__(self, buildings: int = 2) -> None:
        self.summaries: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.buildings = buildings

    def _s(self, k: str, v: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(k)
        self.summaries[k] = v
        return v

    def discover(self) -> dict[str, Any]:
        return self._s("discover", {"city": "Houston", "apartment_communities": 2})

    def sunlight(self) -> dict[str, Any]:
        return self._s("sunlight", {"sun_positions": 100, "latitude": 29.7, "cloud_loss_percent": 20})

    def trace_roofs(self) -> dict[str, Any]:
        return self._s("trace_roofs", {"analyzed": 2, "roofs": 3, "roof_sides": 6, "panels": 50,
                                       "ac_units_left_room_for": 1})

    def rank(self) -> dict[str, Any]:
        return self._s("rank", {"ranked": 2, "total_savings_year": 1000, "built_in_3d": 2})

    def verify(self) -> dict[str, Any]:
        return self._s("verify", {"nrel_check": {"median_abs_diff_pct": 1.9, "rank_agreement": 0.97}})

    def assemble(self) -> dict[str, Any]:
        return {"city": "Houston", "buildings": [{"id": i} for i in range(self.buildings)]}


class Scripted(GenericFakeChatModel):
    """A chat model that replays fixed AI messages (tool calls, then a brief)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "Scripted":
        return self


def scripted(tool_names: list[str], brief: str = "Measured 2 sites; the top saves $1,000 a year.") -> Scripted:
    msgs = [AIMessage(content="", tool_calls=[{"name": "write_todos", "id": "plan", "args": {
        "todos": [{"content": "Find buildings", "status": "pending"}, {"content": "Rank roofs", "status": "pending"}]}}])]
    msgs += [AIMessage(content="", tool_calls=[{"name": n, "id": f"c{i}", "args": {}}]) for i, n in enumerate(tool_names)]
    msgs.append(AIMessage(content=brief))
    return Scripted(messages=iter(msgs))


def run_with(model: Scripted, city: FakeCity) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    out = sc.run_city("houston", 2, emit=events.append, city_run=city, use_agent=True,
                      agent_factory=lambda tools: sc.build_solar_agent(tools, model))
    return out, events


def test_real_deep_agent_plans_and_calls_every_tool_in_order() -> None:
    city = FakeCity()
    out, events = run_with(scripted(TOOL_NAMES), city)
    assert out["run"]["mode"] == "agent"
    assert out["run"]["backend_filled"] == [] and out["run"]["agent_note"] is None
    assert city.calls == ["discover", "sunlight", "trace_roofs", "rank", "verify"]
    done = [e for e in events if e["event"] == "step" and e["status"] == "done"]
    assert [e["tool"] for e in done] == TOOL_NAMES and {e["by"] for e in done} == {"agent"}
    assert any(e["event"] == "plan" and e["todos"] == ["Find buildings", "Rank roofs"] for e in events)
    assert out["run"]["brief"].startswith("Measured 2 sites")
    assert out["data"]["buildings"] == [{"id": 0}, {"id": 1}]


def test_skipped_tools_are_finished_by_the_backend_and_labelled() -> None:
    city = FakeCity()
    out, events = run_with(scripted(TOOL_NAMES[:2]), city)
    assert out["run"]["mode"] == "agent"
    assert out["run"]["backend_filled"] == TOOL_NAMES[2:]
    assert "skipped" in out["run"]["agent_note"]
    by = {e["tool"]: e["by"] for e in events if e["event"] == "step" and e["status"] == "done"}
    assert by == {**{t: "agent" for t in TOOL_NAMES[:2]}, **{t: "backend" for t in TOOL_NAMES[2:]}}


def test_prerequisites_show_as_their_own_steps_when_the_agent_jumps_ahead() -> None:
    city = FakeCity()
    _, events = run_with(scripted(["rank_roofs_by_savings", *TOOL_NAMES]), city)
    done = [e["tool"] for e in events if e["event"] == "step" and e["status"] == "done"]
    assert done == TOOL_NAMES  # each stage once, in pipeline order
    assert city.calls == ["discover", "sunlight", "trace_roofs", "rank", "verify"]


def test_agent_failure_falls_back_to_the_same_tools_and_says_why() -> None:
    def broken(tools: Any) -> Any:
        raise RuntimeError("no model credentials")

    out = sc.run_city("houston", 2, city_run=FakeCity(), use_agent=True, agent_factory=broken)
    assert out["run"]["mode"] == "direct"
    assert "RuntimeError" in out["run"]["agent_note"] and "credentials" not in out["run"]["agent_note"]
    assert [s["tool"] for s in out["run"]["steps"]] == TOOL_NAMES
    assert {s["by"] for s in out["run"]["steps"]} == {"backend"}


def test_offline_mode_never_builds_the_agent() -> None:
    def must_not_run(tools: Any) -> Any:
        raise AssertionError("the model must not be used offline")

    out = sc.run_city("houston", 2, city_run=FakeCity(), use_agent=False, agent_factory=must_not_run)
    assert out["run"]["mode"] == "direct" and "Offline" in out["run"]["agent_note"]
    assert out["run"]["brief"] is None


def test_tools_are_langchain_tools_with_no_arguments() -> None:
    run = sc.SolarCityRun("houston", 2, lambda e: None, city_run=FakeCity())
    tools = run.tools()
    assert [t.name for t in tools] == TOOL_NAMES
    assert all(t.args == {} and t.description for t in tools)
    assert json.loads(tools[0].invoke({}))["apartment_communities"] == 2


def test_failing_stage_reports_type_only_and_raises_tool_error() -> None:
    class Broken(FakeCity):
        def discover(self) -> dict[str, Any]:
            raise ValueError("secret detail https://api.example?key=abc")

    events: list[dict[str, Any]] = []
    run = sc.SolarCityRun("houston", 2, events.append, city_run=Broken())
    out = run.tools()[0].invoke({})  # handle_tool_error returns the message to the agent
    assert "ValueError" in out and "secret" not in out
    err = [e for e in events if e.get("status") == "error"][0]
    assert "ValueError" in err["text"] and "secret" not in err["text"]


def test_step_text_bolds_real_numbers() -> None:
    assert sc.step_text("discover", {"apartment_communities": 43, "city": "Houston"}) == \
        "Read CoStar: **43** apartment communities in Houston"
    assert "**1.9%**" in sc.step_text("verify", {"nrel_check": {"median_abs_diff_pct": 1.9}})
    assert "no NREL result" in sc.step_text("verify", {"nrel_check": {}})


def _frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for frame in text.strip().split("\n\n"):
        lines = frame.splitlines()
        kind = next((line[6:].strip() for line in lines if line.startswith("event:")), None)
        body = "".join(line[5:].strip() for line in lines if line.startswith("data:"))
        if kind and body:
            out.append((kind, json.loads(body)))
    return out


def test_stream_endpoint_sends_steps_then_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.main import app

    real = sc.run_city

    def fake_run(city: str, n: int = 50, emit: Any = None, **kw: Any) -> dict[str, Any]:
        return real(city, n, emit=emit, city_run=FakeCity(), use_agent=False)

    monkeypatch.setattr(sc, "run_city", fake_run)
    with TestClient(app) as client:
        res = client.post("/api/solar-city/stream", json={"city": "houston", "n": 2})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    frames = _frames(res.text)
    kinds = [k for k, _ in frames]
    assert kinds[-1] == "result" and kinds.count("step") == 10
    assert frames[-1][1]["run"]["mode"] == "direct" and len(frames[-1][1]["data"]["buildings"]) == 2


def test_stream_endpoint_reports_failure_without_details(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.main import app

    def boom(*a: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("key=SECRET")

    monkeypatch.setattr(sc, "run_city", boom)
    with TestClient(app) as client:
        res = client.post("/api/solar-city/stream", json={"city": "houston"})
    frames = _frames(res.text)
    assert frames == [("error", {"message": sc.FAILURE_MESSAGE})] and "SECRET" not in res.text


def test_stream_endpoint_rejects_unknown_city() -> None:
    from backend.main import app

    with TestClient(app) as client:
        assert client.post("/api/solar-city/stream", json={"city": "paris"}).status_code == 422


def test_cli_never_overwrites_page_data_with_an_empty_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.build_solar_demo as build

    class Empty:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def assemble(self) -> dict[str, Any]:
            return {"counts": {"analyzed": 0}, "buildings": []}

    monkeypatch.setattr(build, "CityRun", Empty)
    monkeypatch.setattr(build, "PROJECT_ROOT", tmp_path)
    build.main(["--city", "houston"])
    assert not (tmp_path / "frontend" / "solar" / "data_houston.json").exists()


def test_city_run_rejects_unknown_city() -> None:
    from scripts.build_solar_demo import CityRun

    with pytest.raises(ValueError):
        CityRun("paris")


def test_solar_page_streams_from_the_backend_instead_of_timed_fake_steps() -> None:
    js = (ROOT / "frontend" / "solar" / "solar.js").read_text(encoding="utf-8")
    assert 'fetch("/api/solar-city/stream"' in js
    assert "await sleep(650)" not in js  # the old scripted animation is gone
    assert "Showing the recorded run instead" in js  # fallback is labelled, never silent
    html = (ROOT / "frontend" / "solar" / "index.html").read_text(encoding="utf-8")
    assert 'id="thinking"' in html and 'id="steps"' in html  # same UI elements
