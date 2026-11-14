"""POST /api/ask-stream (SSE) and GET /api/image/{id}, offline with a scripted agent."""

import json
import socket
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import config, main
from backend.agent import agent_api, planner
from backend.schemas import AskResponse

QUESTION = "In Houston, do apartment buildings with more parking spaces have higher asking rent?"
SECRET = "private-stream-secret-123"
_REAL_RUN = planner.run_question


class StreamAgent:
    """Delegates (with hand-offs), runs the research tools, writes a cited brief."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = {t.name: t for t in tools}

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        printer = config["callbacks"][0]
        handoff = lambda name, text="": printer.on_tool_start(
            {"name": "task"}, "", inputs={"subagent_type": name, "description": text or name})
        handoff("sampler", f"sample Houston using {SECRET}")  # a leaked secret must be redacted
        s = self.tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11})
        handoff("inspector")
        for b in s["buildings"][:3]:
            self.tools["vexcel_detect"].invoke({"lat": b["latitude"], "lon": b["longitude"]})
            self.tools["vision_inspect"].invoke({"property_id": b["id"]})
        handoff("statistician")
        st = self.tools["run_regression"].invoke({"question": QUESTION})
        handoff("writer")
        p = st["predictor"]
        brief = (f"## Question\n{QUESTION}\n\n## Result\nn = {s['count']} [{s['evidence_id']}], coefficient "
                 f"{st['coefficients'][p]} [{st['evidence_id']}] (demo data). Key {SECRET}.\n")
        return {"messages": [*inputs["messages"], AIMessage(content=brief)]}


def parse(text: str) -> tuple[list[tuple[str, dict]], list[str]]:
    """Parse an SSE body into (event, data) frames plus comment lines."""
    frames, comments = [], []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            comments.append(block)
            continue
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and len(lines) == 2, block
        frames.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return frames, comments


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """No network; the planner runs for real with the scripted agent and mock tools."""
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setenv("VEXCEL_API_KEY", SECRET)

    def fake_run(question: str, mode: str | None = None, n: int = 40, intent: str = "auto",
                 on_event=None) -> planner.PlannerRun:
        return _REAL_RUN(question, mode=mode, n=n, intent=intent, on_event=on_event, agent_factory=StreamAgent)

    monkeypatch.setattr(planner, "run_question", fake_run)
    return TestClient(main.app)


def stream(client: TestClient, body: dict) -> tuple[Any, str]:
    """POST to the stream route and return (response, full body text)."""
    with client.stream("POST", "/api/ask-stream", json=body) as response:
        return response, "".join(response.iter_text())


def test_frames_arrive_in_order_and_parse_as_sse(client: TestClient) -> None:
    """handoff/tool_call frames stream live, then citation_check, evidence and result last."""
    response, text = stream(client, {"question": QUESTION, "use_mock_tools": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache" and response.headers["x-accel-buffering"] == "no"
    frames, _ = parse(text)
    kinds = [k for k, _ in frames]
    assert kinds[0] == "handoff" and kinds[-3:] == ["citation_check", "evidence", "result"]
    assert set(kinds[:-3]) == {"handoff", "tool_call"}
    # Live order: the sampler hand-off precedes sample_buildings, which precedes the inspector hand-off.
    labels = [d["payload"].get("subagent") or d["payload"].get("tool") for k, d in frames if k in ("handoff", "tool_call")]
    assert labels[:3] == ["sampler", "sample_buildings", "inspector"]
    assert labels.index("statistician") < labels.index("run_regression") < labels.index("writer")
    tool_ids = [d["payload"]["evidence_id"] for k, d in frames if k == "tool_call"]
    assert tool_ids == [f"E{i}" for i in range(1, len(tool_ids) + 1)]
    first_tool = next(d for k, d in frames if k == "tool_call")
    assert set(first_tool["payload"]) == {"evidence_id", "tool", "source", "cached", "failed", "property_id"}
    assert frames[-3][1]["message"] == "PASS"


def test_evidence_and_result_frames_are_valid(client: TestClient) -> None:
    """evidence lists every ledger entry with a short summary; result is a valid AskResponse."""
    _, text = stream(client, {"question": QUESTION, "use_mock_tools": True})
    frames = dict(parse(text)[0][-2:])
    evidence = frames["evidence"]["evidence"]
    result = AskResponse.model_validate(frames["result"])
    assert [e["id"] for e in evidence] == [f"E{i}" for i in range(1, len(evidence) + 1)]
    assert all(set(e) == {"id", "tool", "source", "summary"} and e["summary"] for e in evidence)
    vision = next(e for e in evidence if e["tool"] == "vision_inspect")
    assert vision["summary"].startswith("parking_space_count ") and " for property 9900" in vision["summary"]
    regression = next(e for e in evidence if e["tool"] == "run_regression")
    assert regression["summary"].startswith("parking_space_count coefficient")
    assert result.study.stats is not None and result.events[-1].event_type == "citation_check"
    assert len([e for e in result.events if e.event_type == "tool_call"]) == len(evidence)


def test_no_secrets_args_or_raw_responses_in_any_frame(client: TestClient) -> None:
    """Secrets are redacted everywhere; tool arguments and raw Vexcel responses never appear."""
    _, text = stream(client, {"question": QUESTION, "use_mock_tools": True})
    assert SECRET not in text and "[REDACTED]" in text
    frames, _ = parse(text)
    live = "".join(json.dumps(d) for k, d in frames if k != "result")
    assert "raw_response" not in live and '"lat"' not in live and '"args"' not in live
    assert "raw_response" not in json.dumps([e for e in frames[-1][1]["events"]])


def test_planner_failure_yields_one_error_frame_and_closes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception becomes exactly one credential-free error frame, then the stream ends."""
    def broken(question: str, mode=None, n=40, intent="auto", on_event=None):
        on_event({"event_type": "handoff", "message": "hand-off -> sampler", "timestamp": None,
                  "payload": {"subagent": "sampler", "task": "x"}})
        raise RuntimeError(f"provider exploded with {SECRET}")

    monkeypatch.setattr(planner, "run_question", broken)
    response, text = stream(client, {"question": QUESTION, "use_mock_tools": True})
    frames, _ = parse(text)
    assert [k for k, _ in frames] == ["handoff", "error"]
    assert frames[-1][1] == {"message": agent_api.FAILURE_MESSAGE}
    assert SECRET not in text and "exploded" not in text


def test_keepalive_comments_while_waiting(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """': keepalive' comments are sent while the planner is busy."""
    monkeypatch.setattr(agent_api, "KEEPALIVE_SECONDS", 0.05)
    real = planner.run_question

    def slow(*args: Any, **kwargs: Any) -> planner.PlannerRun:
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "run_question", slow)
    _, text = stream(client, {"question": QUESTION, "use_mock_tools": True})
    frames, comments = parse(text)
    assert comments and all(c == ": keepalive" for c in comments)
    assert frames[-1][0] == "result"


def test_mode_and_validation_match_ask_agent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same AskRequest validation; mock/auto mode, capped n and intent pass through."""
    seen: list[dict] = []

    def record(question: str, mode=None, n=40, intent="auto", on_event=None):
        seen.append({"mode": mode, "n": n, "intent": intent, "live": on_event is not None})
        raise RuntimeError("stop")

    monkeypatch.setattr(planner, "run_question", record)
    stream(client, {"question": QUESTION, "limit": 200, "mode": "retrofit"})
    assert seen == [{"mode": "auto", "n": 50, "intent": "retrofit", "live": True}]
    for body in ({"question": ""}, {}, {"question": QUESTION, "limit": 0}, {"question": QUESTION, "extra": 1},
                 {"question": QUESTION, "mode": "other"}):
        assert client.post("/api/ask-stream", json=body).status_code == 422
    assert len(seen) == 1


def test_on_event_none_leaves_planner_unchanged() -> None:
    """Without on_event the planner still returns the same PlannerRun (no listener needed)."""
    run = _REAL_RUN(QUESTION, mode="mock", agent_factory=StreamAgent)
    assert run.complete and run.study.stats is not None


def test_listener_errors_do_not_break_the_run() -> None:
    """A failing on_event listener is ignored; the study still completes."""
    def bad(event: dict) -> None:
        raise ValueError("listener bug")

    run = _REAL_RUN(QUESTION, mode="mock", agent_factory=StreamAgent, on_event=bad)
    assert run.complete


# ---- GET /api/image/{property_id} ------------------------------------------------------

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


@pytest.fixture
def images(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    """Showcase and Vexcel-cache image folders under tmp_path."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "root")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    showcase = tmp_path / "root" / "data" / "showcase" / "images"
    cache = tmp_path / "cache" / "vexcel" / "images"
    showcase.mkdir(parents=True)
    cache.mkdir(parents=True)
    return {"showcase": showcase, "cache": cache, "tmp": tmp_path}


def test_image_served_from_showcase_then_cache(images: dict) -> None:
    """Showcase images win; otherwise the Vexcel cache image is served as JPEG."""
    (images["cache"] / "property_23453.jpg").write_bytes(JPEG + b"cache")
    client = TestClient(main.app)
    response = client.get("/api/image/23453")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.content.endswith(b"cache")
    (images["showcase"] / "property_23453.jpg").write_bytes(JPEG + b"showcase")
    assert client.get("/api/image/23453").content.endswith(b"showcase")


def test_missing_image_is_404(images: dict) -> None:
    """No file for the id: 404."""
    assert TestClient(main.app).get("/api/image/12345").status_code == 404


@pytest.mark.parametrize("path", ["/api/image/..%2F..%2F.env", "/api/image/../../.env", "/api/image/1.jpg",
                                  "/api/image/-1", "/api/image/%2Fetc%2Fpasswd", "/api/image/1%2F..%2F2"])
def test_image_route_rejects_traversal_and_non_integers(images: dict, path: str) -> None:
    """Only non-negative integer ids are accepted; nothing outside the image folders is served."""
    (images["tmp"] / "root" / ".env").write_text(f"VEXCEL_API_KEY={SECRET}")
    response = TestClient(main.app).get(path)
    assert response.status_code in (404, 422)


# ---- GET /api/roof-view/{property_id} --------------------------------------------------

def test_roof_view_served_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """scripts/roof_solar_view.py's report image is served as JPEG when it exists."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    folder = tmp_path / "roof_view"
    folder.mkdir()
    (folder / "property_23453_report.jpg").write_bytes(JPEG)
    response = TestClient(main.app).get("/api/roof-view/23453")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.content == JPEG


def test_roof_view_missing_is_404(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A building scripts/roof_solar_view.py has not processed yet: 404, not an error."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    assert TestClient(main.app).get("/api/roof-view/999999").status_code == 404


@pytest.mark.parametrize("path", ["/api/roof-view/..%2F..%2F.env", "/api/roof-view/-1",
                                  "/api/roof-view/1.jpg", "/api/roof-view/1%2F..%2F2"])
def test_roof_view_rejects_traversal_and_non_integers(monkeypatch: pytest.MonkeyPatch, tmp_path, path: str) -> None:
    """Only non-negative integer ids are accepted; nothing outside data/cache/roof_view is served."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    (tmp_path / ".env").write_text(f"VEXCEL_API_KEY={SECRET}")
    response = TestClient(main.app).get(path)
    assert response.status_code in (404, 422)
    assert SECRET.encode() not in response.content


# ---- GET /api/roof-view/{property_id}/summary -------------------------------------------

def _fake_report(width: int = 2860, height: int = 2855) -> bytes:
    """A plain JPEG the same shape as scripts/roof_solar_view.py's real report canvas."""
    from io import BytesIO

    from PIL import Image

    image = Image.new("RGB", (width, height), (250, 250, 248))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_roof_view_summary_crops_out_the_calculation_text(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The summary image drops the title/caption band and starts at the annotated panels.

    Marks the title band a distinct color from the panel row below it in the fake report,
    so a crop that keeps the top of the image (instead of cutting it off) is caught.
    """
    from io import BytesIO

    from backend.agent import agent_api
    from PIL import Image

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    folder = tmp_path / "roof_view"
    folder.mkdir()
    width, height = 2860, 2855
    image = Image.new("RGB", (width, height), (10, 200, 10))  # panel row: green
    band_h = agent_api._REPORT_TITLE_H + agent_api._REPORT_CAP_H
    image.paste(Image.new("RGB", (width, band_h), (200, 10, 10)), (0, 0))  # title/caption band: red
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    (folder / "property_23453_report.jpg").write_bytes(buffer.getvalue())

    response = TestClient(main.app).get("/api/roof-view/23453/summary")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    cropped = Image.open(BytesIO(response.content))
    assert cropped.width == 2860
    assert cropped.height < 1600  # well short of the full 2855 px report
    r, g, b = cropped.getpixel((width // 2, 30))  # past the crop's small kept margin below the band
    assert g > r  # the row is the panel strip, not the red title band


def test_roof_view_summary_missing_is_404(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    assert TestClient(main.app).get("/api/roof-view/999999/summary").status_code == 404


def test_mock_ids_get_a_placeholder_svg(images: dict) -> None:
    """Mock-cohort ids (e.g. 990101) get a deterministic SVG placeholder, labelled synthetic."""
    client = TestClient(main.app)
    response = client.get("/api/image/990101")
    assert response.status_code == 200 and response.headers["content-type"].startswith("image/svg+xml")
    assert response.text.startswith("<svg") and "synthetic placeholder" in response.text
    assert response.text == client.get("/api/image/990101").text
    assert response.text != client.get("/api/image/990102").text


def test_large_real_ids_are_not_treated_as_mock(images: dict) -> None:
    """Real CoStar ids above 990000 (e.g. 10572052) get their JPEG or 404, never the placeholder."""
    client = TestClient(main.app)
    assert client.get("/api/image/10572052").status_code == 404
    assert client.get("/api/image/990500").status_code == 404  # in the range but not a mock id
    (images["cache"] / "property_10572052.jpg").write_bytes(JPEG)
    response = client.get("/api/image/10572052")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
