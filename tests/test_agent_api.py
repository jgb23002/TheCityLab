"""POST /api/ask-agent with a fake planner agent: no network, keys or model calls."""

import socket
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import main
from backend.agent import agent_api, planner
from backend.schemas import AskResponse

QUESTION = "In Houston, do apartment buildings with more parking spaces have higher asking rent?"


class ScriptedAgent:
    """Runs sampler/inspector/statistician tools, reports hand-offs, writes a cited brief."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = {t.name: t for t in tools}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        printer = config["callbacks"][0]
        printer.on_tool_start({"name": "task"}, "", inputs={"subagent_type": "sampler", "description": "sample"})
        s = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
        printer.on_tool_start({"name": "task"}, "", inputs={"subagent_type": "inspector", "description": "inspect"})
        for b in s["buildings"][:3]:
            self.tools["vexcel_detect"].invoke({"lat": b["latitude"], "lon": b["longitude"]})
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        st = self.tools["run_regression"].invoke({"question": QUESTION})
        p = st["predictor"]
        brief = (f"## Question\n{QUESTION}\n\n## Result\nn = {s['count']} [{s['evidence_id']}], coefficient "
                 f"{st['coefficients'][p]} [{st['evidence_id']}] (demo data).\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Block the network and route the planner through the scripted agent."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    real = planner.run_question
    calls: list[dict[str, Any]] = []

    def fake_run(question: str, mode: str | None = None, n: int = 40) -> planner.PlannerRun:
        calls.append({"mode": mode, "n": n})
        return real(question, mode=mode, n=n, agent_factory=ScriptedAgent)

    monkeypatch.setattr(planner, "run_question", fake_run)
    test_client = TestClient(main.app)
    test_client.calls = calls  # type: ignore[attr-defined]
    return test_client


def test_success_returns_ask_response_with_events(client: TestClient) -> None:
    """The study validates and events cover tool calls, hand-offs and the citation check."""
    response = client.post("/api/ask-agent", json={"question": QUESTION, "use_mock_tools": True, "limit": 200})
    assert response.status_code == 200, response.text
    result = AskResponse.model_validate(response.json())
    assert result.study.question == QUESTION and result.study.is_mock
    assert client.calls == [{"mode": "mock", "n": 50}]  # type: ignore[attr-defined]

    tool_events = [e for e in result.events if e.event_type == "tool_call"]
    assert tool_events[0].message == "sample_buildings [mock]"
    assert tool_events[0].payload["evidence_id"] == "E1"
    assert [e.payload["evidence_id"] for e in tool_events] == [f"E{i}" for i in range(1, len(tool_events) + 1)]
    handoffs = [e.payload["subagent"] for e in result.events if e.event_type == "handoff"]
    assert handoffs == ["sampler", "inspector"]
    last = result.events[-1]
    assert last.event_type == "citation_check" and last.message == "PASS"
    assert last.payload == {"verified": True, "problems": []}
    assert len({e.id for e in result.events}) == len(result.events)
    assert "raw_response" not in response.text.split('"events"', 1)[1]


def test_default_mode_is_auto_and_limit_is_passed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """use_mock_tools false selects auto mode; limit below the cap passes through."""
    seen: list[tuple[str | None, int]] = []
    monkeypatch.setattr(planner, "run_question", lambda q, mode=None, n=40: seen.append((mode, n)) or
                        (_ for _ in ()).throw(RuntimeError("stop")))
    client.post("/api/ask-agent", json={"question": QUESTION, "limit": 12})
    assert seen == [("auto", 12)]


def test_planner_failure_returns_credential_free_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any planner exception becomes a 503 without its private details."""
    def broken(question: str, mode: str | None = None, n: int = 40) -> planner.PlannerRun:
        raise RuntimeError("secret-provider-detail AKIA123")

    monkeypatch.setattr(planner, "run_question", broken)
    response = client.post("/api/ask-agent", json={"question": QUESTION, "use_mock_tools": True})
    assert response.status_code == 503
    assert "secret-provider-detail" not in response.text and "AKIA123" not in response.text


@pytest.mark.parametrize("body", [{"question": ""}, {}, {"question": QUESTION, "limit": 0},
                                  {"question": QUESTION, "extra": 1}, {"question": QUESTION, "mode": "other"}])
def test_request_validation_is_unchanged(client: TestClient, body: dict) -> None:
    """AskRequest validation rejects the same bodies as /api/ask, before the planner runs."""
    assert client.post("/api/ask-agent", json=body).status_code == 422
    assert client.post("/api/ask", json=body).status_code == 422
    assert client.calls == []  # type: ignore[attr-defined]


def test_retrofit_is_not_wired_yet(client: TestClient) -> None:
    """Retrofit is Phase 4, so the agent route rejects it clearly instead of running research."""
    response = client.post("/api/ask-agent", json={"question": QUESTION, "mode": "retrofit"})
    assert response.status_code == 422 and client.calls == []  # type: ignore[attr-defined]


def test_events_redact_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secret values in hand-off text or problems never reach event payloads."""
    from backend.agent.evidence import Ledger
    from backend.schemas import StudyResult

    monkeypatch.setenv("VEXCEL_API_KEY", "private-test-value")
    study = StudyResult(id="s1", question="q", buildings=[], measurements=[], stats=None,
                        brief_markdown="", caveats=[])
    run = planner.PlannerRun(study=study, ledger=Ledger(), verified=False,
                             problems=["private-test-value leaked"],
                             handoffs=[{"subagent": "writer", "task": "use private-test-value", "ts": "t"}])
    text = "".join(e.model_dump_json() for e in agent_api.events_from_run(run))
    assert "private-test-value" not in text and "[REDACTED]" in text
