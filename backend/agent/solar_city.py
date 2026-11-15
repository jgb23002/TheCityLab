"""The solar page's live run: a DeepAgents agent ranks a city's apartment roofs for solar.

POST /api/solar-city/stream {"city": "houston" | "los_angeles", "n": 1-50} streams SSE:

- ``plan``   the agent's to-do list (DeepAgents / LangChain ``write_todos``)
- ``step``   each tool as it starts and finishes, with the real numbers it produced
- ``brief``  the agent's short written summary (absent when the model was not used)
- ``result`` the page data (same shape as ``frontend/solar/data_<city>.json``) and run info
- ``error``  one credential-free message, then the stream closes

Each tool runs one stage of ``scripts/build_solar_demo.CityRun`` (CoStar, NOAA/NASA sun,
Vexcel roofs, NREL PVWatts). The model only decides the order and writes the summary; every
number on the page comes from those tools, never from the model.

With ``USE_CACHE_ONLY=true`` no model or provider call is made: the backend runs the same
tools in their fixed order and says so (``mode: "direct"``). If the agent skips a stage, the
backend finishes it and marks that step ``by: "backend"``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from backend import config
from backend.agent.agent_api import KEEPALIVE_SECONDS, SSE_HEADERS, sse_frame

logger = logging.getLogger(__name__)
router = APIRouter()

Emit = Callable[[dict[str, Any]], None]
FAILURE_MESSAGE = "The solar run failed before it finished. Try again, or switch city."

SYSTEM_PROMPT = """You are CityLab's solar analyst. Your job: rank every sampled apartment \
community's roof in one city by what a rooftop solar install would be worth, so an owner can be told.

You have five tools. Each one runs one real stage of the analysis on CoStar, Vexcel, NOAA/NASA \
and NREL data and returns its numbers:
1. find_apartment_buildings - CoStar sample of apartment communities, lot sizes, owners, power price
2. measure_year_of_sun - sun every 10 minutes for a year plus NASA cloud cover at this city
3. trace_roofs_and_place_panels - Vexcel roof outlines, roof sides, real-size panels, yearly savings
4. rank_roofs_by_savings - rank the sites, choose each owner email recipient, build the top 5 in 3D
5. check_against_nrel - compare the sun model with NREL PVWatts

First write a short to-do list with write_todos. Then call all five tools, in order, once each. \
Do not call a tool again with the same input. Finally reply with a brief of at most 4 sentences \
for a property owner: how many roofs were measured, the top site and its yearly savings and \
payback, and how far the model is from NREL. Use only numbers the tools returned; never invent one."""

# tool name -> (CityRun stage, label shown on the page)
TOOLS: dict[str, tuple[str, str]] = {
    "find_apartment_buildings": ("discover", "Reading CoStar apartment communities"),
    "measure_year_of_sun": ("sunlight", "Following the sun for a year, with NASA weather"),
    "trace_roofs_and_place_panels": ("trace_roofs", "Tracing roofs from Vexcel and placing real-size panels"),
    "rank_roofs_by_savings": ("rank", "Ranking roofs by yearly savings, building the top 5 in 3D"),
    "check_against_nrel": ("verify", "Checking the sun model against NREL PVWatts"),
}
TOOL_DESCRIPTIONS = {
    "find_apartment_buildings": "Sample the city's apartment communities from CoStar with lot sizes, owners and the local power price.",
    "measure_year_of_sun": "Compute the sun position every 10 minutes for a year and NASA POWER cloud cover at the sample centre.",
    "trace_roofs_and_place_panels": "Match each Vexcel parcel, split roofs into sides, fit real-size panels with a 3 ft fire gap, and price each site.",
    "rank_roofs_by_savings": "Rank sites by yearly savings, pick each owner email recipient and build the top 5 roofs in 3D.",
    "check_against_nrel": "Compare CityLab's sun model with NREL PVWatts for the same roofs and summarize the evidence.",
}


class NoArgs(BaseModel):
    """These tools take no arguments: the city is fixed for the run."""


class SolarCityRequest(BaseModel):
    """Body of POST /api/solar-city/stream."""

    city: Literal["houston", "los_angeles"] = "houston"
    n: int = Field(50, ge=1, le=50, description="apartment communities to sample")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def step_text(stage: str, s: dict[str, Any]) -> str:
    """One line for the page, with the stage's real numbers in **bold**."""
    if stage == "discover":
        return f"Read CoStar: **{s['apartment_communities']}** apartment communities in {s['city']}"
    if stage == "sunlight":
        cloud = s.get("cloud_loss_percent")
        tail = f"; NASA: clouds block **{cloud}%** of the sun" if cloud is not None else ""
        return f"Followed the sun: **{s['sun_positions']:,}** sun positions at {s['latitude']}°N{tail}"
    if stage == "trace_roofs":
        return (f"Traced **{s['roofs']:,}** roofs and **{s['roof_sides']:,}** roof sides on **{s['analyzed']}** sites; "
                f"placed **{s['panels']:,}** panels, leaving room for **{s['ac_units_left_room_for']:,}** AC units")
    if stage == "rank":
        return (f"Ranked **{s['ranked']}** sites: **${s['total_savings_year']:,}** a year in total; "
                f"built the top **{s['built_in_3d']}** in 3D")
    if stage == "verify":
        n = s.get("nrel_check") or {}
        diff, agree = n.get("median_abs_diff_pct"), n.get("rank_agreement")
        if diff is None:
            return "Checked against NREL PVWatts: no NREL result available for these roofs"
        return (f"Checked against NREL PVWatts: roof sides differ by **{diff}%** (median)"
                + (f", rank agreement **{agree}**" if agree is not None else ""))
    return stage


class SolarCityRun:
    """A CityRun exposed as LangChain tools that report each step as it happens."""

    def __init__(self, city: str, n: int, emit: Emit, city_run: Any | None = None) -> None:
        if city_run is None:
            from scripts.build_solar_demo import CityRun  # heavy imports only when a run starts
            city_run = CityRun(city, n=n)
        self.run, self.emit = city_run, emit
        self.lock = threading.RLock()  # the agent may call tools in parallel; stages share state
        self.steps: list[dict[str, Any]] = []
        self.done_by: dict[str, str] = {}

    def call(self, tool: str, by: str) -> dict[str, Any]:
        """Run one stage (its prerequisites first), emitting start and done/error steps."""
        stage, label = TOOLS[tool]
        with self.lock:
            if stage in self.done_by:
                return self.run.summaries[stage]
            for earlier, (st, _) in TOOLS.items():  # prerequisites show as their own steps
                if earlier == tool:
                    break
                if st not in self.done_by:
                    self.call(earlier, by)
            self.emit({"event": "step", "tool": tool, "label": label, "status": "start", "by": by, "timestamp": _now()})
            try:
                summary = getattr(self.run, stage)()
            except Exception as exc:  # noqa: BLE001 - report the type, never provider text
                logger.exception("solar stage %s failed", stage)
                self.emit({"event": "step", "tool": tool, "label": label, "status": "error", "by": by,
                           "text": f"{label} failed ({type(exc).__name__})", "timestamp": _now()})
                raise ToolException(f"{tool} failed ({type(exc).__name__}); do not retry it") from exc
            self.done_by[stage] = by
            record = {"event": "step", "tool": tool, "label": label, "status": "done", "by": by,
                      "text": step_text(stage, summary), "summary": summary, "timestamp": _now()}
            self.steps.append(record)
            self.emit(record)
            return summary

    def tools(self) -> list[StructuredTool]:
        """The five stages as LangChain tools for the agent."""
        def make(name: str) -> StructuredTool:
            def fn() -> str:
                return json.dumps(self.call(name, "agent"), default=str)
            return StructuredTool.from_function(func=fn, name=name, description=TOOL_DESCRIPTIONS[name],
                                                args_schema=NoArgs, handle_tool_error=True)
        return [make(name) for name in TOOLS]

    def finish(self) -> list[str]:
        """Run any stage the agent skipped, in order; return the tools the backend had to run."""
        filled = []
        for tool, (stage, _) in TOOLS.items():
            if stage not in self.done_by:
                self.call(tool, "backend")
                filled.append(tool)
        return filled


class _PlanListener(BaseCallbackHandler):
    """Forward the agent's write_todos plan to the page."""

    raise_error = False

    def __init__(self, emit: Emit) -> None:
        super().__init__()
        self.emit = emit

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *,
                      inputs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if (serialized or {}).get("name") != "write_todos":
            return
        if inputs is None:
            try:
                inputs = json.loads(input_str)
            except (TypeError, ValueError):
                inputs = {}
        todos = [str(t.get("content", "")) for t in (inputs or {}).get("todos") or [] if isinstance(t, dict)]
        self.emit({"event": "plan", "todos": [t for t in todos if t][:8], "timestamp": _now()})


def build_solar_agent(tools: list[StructuredTool], model: Any) -> Any:
    """The DeepAgents orchestrator for the solar page (create_deep_agent + LangChain's write_todos)."""
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware

    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
                             middleware=[TodoListMiddleware()], name="CityLab Solar")


def _default_agent(tools: list[StructuredTool]) -> Any:
    from backend.llm import get_chat_model

    return build_solar_agent(tools, get_chat_model("chat"))


def _final_text(messages: list[Any]) -> str | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            text = m.content if isinstance(m.content, str) else " ".join(
                p.get("text", "") for p in m.content if isinstance(p, dict))
            text = text.strip()
            if text:
                return text
    return None


def run_city(city: str, n: int = 50, emit: Emit | None = None,
             agent_factory: Callable[[list[StructuredTool]], Any] | None = None,
             city_run: Any | None = None, use_agent: bool | None = None) -> dict[str, Any]:
    """Run the solar ranking for *city* and return {"data", "run"} (blocking).

    use_agent defaults to ``not USE_CACHE_ONLY``. When the agent cannot start or fails, the
    backend runs the remaining tools itself and says why in ``run.agent_note``.
    """
    events: list[dict[str, Any]] = []

    def send(e: dict[str, Any]) -> None:
        events.append(e)
        if emit:
            try:
                emit(e)
            except Exception:  # noqa: BLE001 - a slow or gone client must not stop the run
                pass

    use_agent = (not config.USE_CACHE_ONLY) if use_agent is None else use_agent
    solar_run = SolarCityRun(city, n, send, city_run=city_run)
    mode, note, brief = "direct", None, None
    if use_agent:
        try:
            agent = (agent_factory or _default_agent)(solar_run.tools())
            send({"event": "agent", "message": "CityLab's DeepAgents planner is choosing the steps", "timestamp": _now()})
            question = (f"Rank the apartment roofs in {solar_run.run.summaries.get('city') or city.replace('_', ' ').title()} "
                        f"(sample of {n}) by what solar would be worth.")
            state = agent.invoke({"messages": [HumanMessage(content=question)]},
                                 config={"callbacks": [_PlanListener(send)], "recursion_limit": 60,
                                         "run_name": "citylab-solar-city", "tags": ["citylab", "solar-page", city]})
            mode, brief = "agent", _final_text(list(state.get("messages") or []))
        except Exception as exc:  # noqa: BLE001 - fall back to the same tools, and say so
            logger.exception("solar agent failed")
            note = f"The agent stopped early ({type(exc).__name__}); the backend ran the remaining steps itself."
    else:
        note = "Offline mode (USE_CACHE_ONLY): no model call; the backend ran the same tools in order."
    filled = solar_run.finish()
    if mode == "agent" and filled:
        note = "The agent skipped " + ", ".join(filled) + "; the backend ran it to complete the analysis."
    if brief:
        send({"event": "brief", "text": brief, "timestamp": _now()})
    data = solar_run.run.assemble()
    run_info = {"mode": mode, "agent_note": note, "brief": brief, "backend_filled": filled,
                "steps": [{k: s[k] for k in ("tool", "label", "by", "text")} for s in solar_run.steps],
                "analyzed": len(data.get("buildings") or []), "cache_only": config.USE_CACHE_ONLY,
                "generated_at": _now()}
    return {"data": data, "run": run_info}


@router.post("/api/solar-city/stream")
async def solar_city_stream(body: SolarCityRequest, http_request: Request) -> StreamingResponse:
    """Stream the live solar ranking for one city (see module docstring for the events)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def put(kind: str, item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, item))

    def worker() -> None:
        try:
            result = run_city(body.city, body.n, emit=lambda e: put("event", e))
        except Exception:  # noqa: BLE001 - message must stay credential-free
            logger.exception("solar city run failed")
            put("error", None)
            return
        put("done", result)

    loop.run_in_executor(None, worker)

    async def frames() -> AsyncIterator[str]:
        while True:
            try:
                kind, item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await http_request.is_disconnected():
                    return
                yield ": keepalive\n\n"
                continue
            if kind == "event":
                yield sse_frame(item["event"], {k: v for k, v in item.items() if k != "event"})
            elif kind == "error":
                yield sse_frame("error", {"message": FAILURE_MESSAGE})
                return
            else:
                yield sse_frame("result", item)
                return

    return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)
