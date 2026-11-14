"""Agent API: POST /api/ask-agent, POST /api/ask-stream (SSE) and GET /api/image/{id}.

The planner is blocking, so it runs in a worker thread. Events are built from the
evidence ledger (tool calls), the recorded subagent hand-offs, and the citation
check (see backend/agent/events.py). They carry ids, tool names and sources only:
no arguments, secrets or raw Vexcel responses.
"""

from __future__ import annotations

import asyncio
import json
import zlib
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Path as PathParam, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend import config
from backend.agent import planner
from backend.agent.events import citation_event, evidence_list, handoff_event, tool_call_event
from backend.agent.registry import redact
from backend.schemas import AgentEvent, AskRequest, AskResponse

MAX_AGENT_N = 50
KEEPALIVE_SECONDS = 15.0
MOCK_ID_MIN = 990_000  # mock cohort ids start here (see _is_mock_id)
FAILURE_MESSAGE = ("The agent could not complete the study. "
                   "Check model settings, tool availability and cached results.")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

router = APIRouter()


def events_from_run(run: planner.PlannerRun) -> list[AgentEvent]:
    """Build tool_call and handoff events in time order, then one citation_check event."""
    run_id = run.study.id
    timeline = [tool_call_event(e) for e in run.ledger.all()] + [handoff_event(h) for h in run.handoffs]
    timeline.sort(key=lambda event: event["timestamp"] or "")  # stable: equal timestamps keep order
    timeline.append(citation_event(run.verified, run.complete, run.problems))
    return [AgentEvent(id=f"{run_id}-ev{i}", run_id=run_id, event_type=e["event_type"], message=e["message"],
                       timestamp=e["timestamp"] or None, payload=e["payload"])
            for i, e in enumerate(timeline, start=1)]


def _planner_kwargs(request: AskRequest) -> dict[str, Any]:
    """Tool mode, capped sample size and intent for one request."""
    return {"mode": "mock" if request.use_mock_tools else "auto",
            "n": min(request.limit, MAX_AGENT_N), "intent": request.mode}


def _run(request: AskRequest) -> planner.PlannerRun:
    """Call the planner with the request's question, intent, capped sample size and tool mode."""
    return planner.run_question(request.question, **_planner_kwargs(request))


@router.post("/api/ask-agent", response_model=AskResponse)
async def ask_agent(request: AskRequest) -> AskResponse:
    """Answer a research or retrofit (rooftop solar) question with the Deep Agents planner."""
    try:
        run = await run_in_threadpool(_run, request)
        return AskResponse(study=run.study, events=events_from_run(run))
    except Exception:
        # Planner/provider exception text can contain request details; never echo it.
        raise HTTPException(status_code=503, detail=FAILURE_MESSAGE) from None


# ---------------------------------------------------------------------------
# Server-sent events
# ---------------------------------------------------------------------------

def sse_frame(event_type: str, data: Any) -> str:
    """One SSE frame: 'event: <type>\\ndata: <json>\\n\\n' (JSON on a single line)."""
    return f"event: {event_type}\ndata: {json.dumps(redact(data), separators=(',', ':'), default=str)}\n\n"


@router.post("/api/ask-stream")
async def ask_stream(request: AskRequest, http_request: Request) -> StreamingResponse:
    """Stream handoff, tool_call and citation_check events live, then evidence and result.

    A failure sends one error frame and closes the stream. A ': keepalive' comment is
    sent every KEEPALIVE_SECONDS while waiting. If the client disconnects, forwarding
    stops (the run may finish in the background).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def put(kind: str, item: Any) -> None:
        """Hand an item from the worker thread to the event loop, in order."""
        loop.call_soon_threadsafe(queue.put_nowait, (kind, item))

    def worker() -> None:
        """Run the blocking planner, feeding live events into the queue."""
        try:
            run = planner.run_question(request.question, **_planner_kwargs(request),
                                       on_event=lambda event: put("event", event))
        except Exception:  # noqa: BLE001 - the message must stay credential-free
            put("error", None)
            return
        put("done", run)

    loop.run_in_executor(None, worker)

    async def frames() -> AsyncIterator[str]:
        """Forward queued items as SSE frames until the result, an error or a disconnect."""
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await http_request.is_disconnected():
                    return
                yield ": keepalive\n\n"
                continue
            if await http_request.is_disconnected():
                return
            if kind == "event":
                yield sse_frame(item["event_type"], {"message": item["message"], "timestamp": item["timestamp"],
                                                     "payload": item["payload"]})
            elif kind == "error":
                yield sse_frame("error", {"message": FAILURE_MESSAGE})
                return
            else:
                try:
                    evidence = evidence_list(item.ledger.all())
                    result = AskResponse(study=item.study, events=events_from_run(item)).model_dump(mode="json")
                except Exception:  # noqa: BLE001
                    yield sse_frame("error", {"message": FAILURE_MESSAGE})
                    return
                yield sse_frame("evidence", {"evidence": evidence})
                yield sse_frame("result", result)
                return

    return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# Building images
# ---------------------------------------------------------------------------

def placeholder_svg(property_id: int) -> str:
    """A deterministic, aerial-style placeholder (roof, parking lot, trees) for mock ids."""
    seed = zlib.crc32(str(property_id).encode())
    roof_w, roof_h = 120 + seed % 60, 80 + (seed >> 6) % 50
    roof_x, roof_y = 40 + (seed >> 12) % 40, 40 + (seed >> 16) % 30
    lot_y = roof_y + roof_h + 30
    stalls = "".join(f'<rect x="{60 + i * 22}" y="{lot_y + 6}" width="2" height="34" fill="#f5f5f5"/>'
                     for i in range(9))
    trees = "".join(
        f'<circle cx="{260 + ((seed >> (i * 3)) % 60)}" cy="{40 + i * 45}" r="{14 + (seed >> i) % 8}" fill="#2f6b2f"/>'
        for i in range(5))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="300" viewBox="0 0 360 300" role="img" '
        f'aria-label="Placeholder aerial image for demo property {property_id}">'
        '<rect width="360" height="300" fill="#7a9a5a"/>'
        f'<rect x="{roof_x}" y="{roof_y}" width="{roof_w}" height="{roof_h}" fill="#b9b4aa" stroke="#6d6a63" stroke-width="3"/>'
        f'<rect x="{roof_x + 15}" y="{roof_y + 15}" width="18" height="12" fill="#8e8a82"/>'
        f'<rect x="50" y="{lot_y}" width="210" height="46" fill="#5b5d60"/>{stalls}{trees}'
        '<rect x="0" y="282" width="360" height="18" fill="#000" fill-opacity="0.45"/>'
        f'<text x="8" y="295" font-family="sans-serif" font-size="11" fill="#fff">DEMO property {property_id} '
        '(synthetic placeholder, not imagery)</text></svg>'
    )


def _is_mock_id(property_id: int) -> bool:
    """True only for ids in the synthetic mock cohort (990001-990012, 990101-990112).

    A plain ">= 990000" rule is wrong: real CoStar ids such as 1146902 or 10572052
    are larger, so the mock cohort itself is checked.
    """
    if property_id < MOCK_ID_MIN:
        return False
    from backend.tools import mock

    return mock.get_property(property_id) is not None


@router.get("/api/image/{property_id}")
def building_image(property_id: int = PathParam(ge=0)) -> Response:
    """Serve a building's aerial JPEG (showcase, then Vexcel cache) or a mock placeholder SVG.

    Real image files are checked first; a placeholder SVG is returned only for ids in
    the mock cohort; anything else is 404. The id is an integer, so the file name cannot
    contain a path; the resolved path is also checked to stay inside its directory.
    """
    name = f"property_{property_id}.jpg"
    for directory in (config.PROJECT_ROOT / "data" / "showcase" / "images",
                      config.CACHE_DIR / "vexcel" / "images"):
        root = directory.resolve()
        path = (root / name).resolve()
        if path.parent == root and path.is_file():
            return FileResponse(path, media_type="image/jpeg")
    if _is_mock_id(property_id):
        return Response(placeholder_svg(property_id), media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=3600"})
    raise HTTPException(status_code=404, detail="No image for this property.")
