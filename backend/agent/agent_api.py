"""POST /api/ask-agent: run the Deep Agents planner and return an AskResponse.

The planner is blocking, so it runs in a worker thread. Events are built from the
evidence ledger (tool calls), the recorded subagent hand-offs, and the citation
check. They carry ids, tool names and sources only: no arguments, secrets or raw
Vexcel responses.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.agent import planner
from backend.agent.registry import redact
from backend.schemas import AgentEvent, AskRequest, AskResponse

MAX_AGENT_N = 50

router = APIRouter()


def events_from_run(run: planner.PlannerRun) -> list[AgentEvent]:
    """Build tool_call and handoff events in time order, then one citation_check event."""
    run_id = run.study.id
    timeline: list[tuple[str, str, str, dict]] = []
    for entry in run.ledger.all():
        payload = {"evidence_id": entry["id"], "tool": entry["tool"], "source": entry["source"],
                   "cached": bool(entry.get("cached")), "failed": "error" in entry["result"]}
        timeline.append((entry["ts"], "tool_call", f"{entry['tool']} [{entry['source']}]", payload))
    for handoff in run.handoffs:
        subagent = str(redact(handoff.get("subagent", "")))
        timeline.append((handoff.get("ts", ""), "handoff", f"hand-off -> {subagent}",
                         {"subagent": subagent, "task": str(redact(handoff.get("task", "")))[:100]}))
    timeline.sort(key=lambda item: item[0])  # stable: equal timestamps keep their order

    events = [AgentEvent(id=f"{run_id}-ev{i}", run_id=run_id, event_type=kind, message=message,
                         timestamp=ts or None, payload=payload)
              for i, (ts, kind, message, payload) in enumerate(timeline, start=1)]
    payload = {"verified": run.verified, "problems": [str(redact(p)) for p in run.problems]}
    if not run.complete:
        message = "FAIL (study incomplete)"
        payload.update({"reason": planner.REGRESSION_INCOMPLETE if any(
            p == planner.REGRESSION_INCOMPLETE for p in run.problems) else "study incomplete"})
    else:
        message = "PASS" if run.verified else "FAIL"
    events.append(AgentEvent(id=f"{run_id}-ev{len(events) + 1}", run_id=run_id,
                             event_type="citation_check", message=message, payload=payload))
    return events


def _run(request: AskRequest) -> planner.PlannerRun:
    """Call the planner with the request's question, capped sample size and tool mode."""
    return planner.run_question(request.question, mode="mock" if request.use_mock_tools else "auto",
                                n=min(request.limit, MAX_AGENT_N))


@router.post("/api/ask-agent", response_model=AskResponse)
async def ask_agent(request: AskRequest) -> AskResponse:
    """Answer a research question with the Deep Agents planner and its cited brief."""
    if request.mode != "research":
        raise HTTPException(status_code=422, detail="The agent route supports research questions only.")
    try:
        run = await run_in_threadpool(_run, request)
        return AskResponse(study=run.study, events=events_from_run(run))
    except Exception:
        # Planner/provider exception text can contain request details; never echo it.
        raise HTTPException(status_code=503, detail="The agent could not complete the study. "
                            "Check model settings, tool availability and cached results.") from None
