"""Offline showcase: build helpers, normalised lookup, replay (sockets disabled), images, size guard."""

import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from PIL import Image

from backend import config, main
from backend.agent import agent_api, planner, showcase
from backend.schemas import AskResponse, CorrectResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_showcase as bs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
QUESTION = "In Houston, do apartment communities with more green cover have higher asking rents?"
_REAL_RUN = planner.run_question


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any socket use: replay must never touch the network."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted during replay")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(agent_api, "REPLAY_DELAY_SECONDS", 0)


class Agent:
    """Fake orchestrator over mock tools (with a fake raw Vexcel response to strip)."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = {t.name: t for t in tools}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        printer = config["callbacks"][0]
        printer.on_tool_start({"name": "task"}, "", inputs={"subagent_type": "sampler", "description": "sample"})
        s = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
        for b in s["buildings"]:
            self.tools["vexcel_detect"].invoke({"lat": b["latitude"], "lon": b["longitude"]})
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        st = self.tools["run_regression"].invoke({"question": QUESTION})
        p, e = st["predictor"], st["evidence_id"]
        brief = (f"## Question\n{QUESTION}\n\n## Result\nThe coefficient is {st['coefficients'][p]} [{e}], "
                 f"n = {st['sample_size']} [{e}] (demo data).\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


@pytest.fixture
def fake_showcase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A one-question showcase in tmp_path built with the real payload/index helpers."""
    root = tmp_path / "repo"
    (root / "data" / "showcase").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setenv("VEXCEL_API_KEY", "private-showcase-secret")
    question = {"id": "Q1", "mode": "research", "question": QUESTION}
    run = _REAL_RUN(QUESTION, mode="mock", agent_factory=Agent)
    run.ledger.all()[1]["result"]["raw_response"] = {"secret": "private-showcase-secret", "big": [1] * 50}
    bs.write_json(root / "data" / "showcase" / "results" / "q1.json", bs.result_payload(question, run))
    bs.write_json(root / "data" / "showcase" / "index.json", bs.build_index([question]))
    return {"root": root, "run": run, "question": question}


def frames(text: str) -> list[tuple[str, dict]]:
    """Parse SSE text into (event, data) pairs."""
    out = []
    for block in text.split("\n\n"):
        if block.startswith("event: "):
            head, data = block.split("\n", 1)
            out.append((head[len("event: "):], json.loads(data[len("data: "):])))
    return out


# ---- build helpers ---------------------------------------------------------------------

def test_payload_excludes_raw_responses_and_secrets(fake_showcase: dict) -> None:
    text = (fake_showcase["root"] / "data" / "showcase" / "results" / "q1.json").read_text()
    assert "raw_response" not in text and "private-showcase-secret" not in text
    saved = json.loads(text)
    assert set(saved) >= {"study", "ledger", "events", "evidence", "verified", "complete"}
    assert saved["events"][-1]["event_type"] == "citation_check"


@pytest.mark.parametrize("variant", [
    QUESTION, QUESTION.upper(), f"   {QUESTION}   ", QUESTION.replace("?", ""),
    "in houston do apartment communities with more green cover have higher asking rents",
    "In Houston, do apartment communities with more green-cover have higher asking rents!!",
])
def test_normalised_matching(fake_showcase: dict, variant: str) -> None:
    assert showcase.lookup(variant) is not None
    assert showcase.normalize_question(variant) == showcase.normalize_question(QUESTION)


def test_unrelated_question_does_not_match(fake_showcase: dict) -> None:
    assert showcase.lookup("In Houston, do apartments with more trees have higher rents?") is None


def test_copy_image_resizes_and_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "cache" / "vexcel" / "images" / "property_42.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (1545, 1300), (90, 120, 80)).save(source, "JPEG", quality=95)
    assert bs.copy_image(42) == "copied"
    with Image.open(tmp_path / "repo" / "data" / "showcase" / "images" / "property_42.jpg") as img:
        assert max(img.size) == 1024 and img.size[0] > img.size[1]
    assert bs.copy_image(42) == "kept" and bs.copy_image(7) == "missing"


def test_size_guard(fake_showcase: dict) -> None:
    size = bs.check_size()
    assert size > 0
    with pytest.raises(SystemExit, match="over the"):
        bs.check_size(limit=size - 1)


# ---- serving ---------------------------------------------------------------------------

def test_cache_only_replays_ask_agent_offline_and_correction_works(
    fake_showcase: dict, offline: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    monkeypatch.setattr(planner, "run_question", lambda *a, **k: pytest.fail("planner must not run in replay"))
    client = TestClient(main.app)
    response = client.post("/api/ask-agent", json={"question": QUESTION.upper(), "use_mock_tools": True})
    assert response.status_code == 200, response.text
    body = AskResponse.model_validate(response.json())
    saved = json.loads((fake_showcase["root"] / "data" / "showcase" / "results" / "q1.json").read_text())
    assert body.study.id == saved["study"]["id"] and len(body.events) == len(saved["events"])
    mid = body.study.measurements[0].id
    corrected = client.post("/api/correct", json={"study_id": body.study.id, "measurement_id": mid,
                                                  "corrections": {"parking_space_count": 3}})
    assert corrected.status_code == 200, corrected.text
    assert CorrectResponse.model_validate(corrected.json()).study.id == f"{body.study.id}-c1"


def test_replay_stream_sends_saved_events_with_delay_then_evidence_and_result(
    fake_showcase: dict, offline: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    sleeps: list[float] = []
    real_sleep = agent_api.asyncio.sleep

    async def spy(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(agent_api, "REPLAY_DELAY_SECONDS", 0.15)
    monkeypatch.setattr(agent_api.asyncio, "sleep", spy)
    with TestClient(main.app).stream("POST", "/api/ask-stream", json={"question": QUESTION}) as response:
        assert response.status_code == 200
        parsed = frames("".join(response.iter_text()))
    saved = json.loads((fake_showcase["root"] / "data" / "showcase" / "results" / "q1.json").read_text())
    kinds = [k for k, _ in parsed]
    assert kinds[:-2] == [e["event_type"] for e in saved["events"]] and kinds[-2:] == ["evidence", "result"]
    assert sleeps == [0.15] * len(saved["events"])
    assert parsed[-2][1]["evidence"] == saved["evidence"]
    assert AskResponse.model_validate(parsed[-1][1]).study.id == saved["study"]["id"]


def test_replay_query_parameter_without_cache_only(fake_showcase: dict, offline: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """replay=true replays a showcase question; without it the planner runs as usual."""
    monkeypatch.setattr(config, "USE_CACHE_ONLY", False)
    calls: list[str] = []

    def live(question: str, **kwargs: Any) -> planner.PlannerRun:
        calls.append(question)
        raise RuntimeError("live run")

    monkeypatch.setattr(planner, "run_question", live)
    client = TestClient(main.app)
    assert client.post("/api/ask-agent?replay=true", json={"question": QUESTION}).status_code == 200
    assert calls == []
    assert client.post("/api/ask-agent", json={"question": QUESTION}).status_code == 503 and calls == [QUESTION]
    # replay=true with a non-showcase question (not cache-only) still runs live.
    client.post("/api/ask-agent?replay=true", json={"question": "Something else in Houston?"})
    assert calls[-1] == "Something else in Houston?"


@pytest.mark.parametrize("route", ["/api/ask-agent", "/api/ask-stream"])
def test_non_showcase_question_in_cache_only_is_a_clear_503(
    fake_showcase: dict, offline: None, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    response = TestClient(main.app).post(route, json={"question": "Do taller buildings in Houston rent for more?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Demo mode: this question is not in the offline showcase."


def test_image_route_serves_showcase_images(fake_showcase: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "empty-cache")
    folder = fake_showcase["root"] / "data" / "showcase" / "images"
    folder.mkdir(parents=True)
    Image.new("RGB", (64, 64), (1, 2, 3)).save(folder / "property_1234567.jpg", "JPEG")
    client = TestClient(main.app)
    response = client.get("/api/image/1234567")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert client.get("/api/image/7654321").status_code == 404


# ---- the committed showcase (Q1-Q3), replayed with sockets disabled ------------------

QUESTIONS = json.loads((REPO / "data" / "showcase" / "questions.json").read_text())["questions"]


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_committed_showcase_replays_each_question_offline(q: dict, offline: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", REPO)
    monkeypatch.setattr(config, "USE_CACHE_ONLY", True)
    client = TestClient(main.app)
    response = client.post("/api/ask-agent", json={"question": q["question"], "mode": q["mode"]})
    assert response.status_code == 200, response.text
    study = AskResponse.model_validate(response.json()).study
    assert study.question == q["question"] and not study.is_mock and study.buildings
    with client.stream("POST", "/api/ask-stream", json={"question": q["question"], "mode": q["mode"]}) as stream:
        kinds = [k for k, _ in frames("".join(stream.iter_text()))]
    assert kinds[-2:] == ["evidence", "result"] and "citation_check" in kinds
    if q["mode"] == "retrofit":
        assert study.solar_results and study.outreach_drafts and study.stats is None
    else:
        assert study.stats is not None


def test_committed_showcase_images_exist_and_fit_the_size_limit() -> None:
    root = REPO / "data" / "showcase"
    images = list((root / "images").glob("property_*.jpg"))
    assert images
    for path in images[:10]:
        with Image.open(path) as img:
            assert max(img.size) <= bs.MAX_IMAGE_PX
    size = sum(p.stat().st_size for p in [root / "index.json", *(root / "results").glob("*.json"), *images])
    assert size <= bs.MAX_BYTES
