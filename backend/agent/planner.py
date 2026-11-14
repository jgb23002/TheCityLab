"""Build and run the CityLab Deep Agent: orchestrator plus four subagents, cited brief.

    python -m backend.agent.planner "<question>" [--mode auto|real|mock] [--n 40] [--json path]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import JsonValue

from backend.agent import prompts
from backend.agent.citations import verify_brief
from backend.agent.events import (
    REGRESSION_INCOMPLETE, citation_event, handoff_event, note_event, plan_event, tool_call_event,
)
from backend.agent.evidence import Ledger
from backend.agent.registry import (
    DEFAULT_N, MAX_N, MODES, detect_intent, draft_ok, get_tools, parse_top_n, ranking_ok, redact,
    regression_ok, resolve_mode, short_args,
)
from backend.llm import get_chat_model
from backend.schemas import Building, Measurement, OutreachDraft, SolarResult, StatsResult, StudyResult

INTENT_CHOICES = ("auto", "research", "retrofit", "property")

SUBAGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "sampler": ("sample_buildings",),
    "inspector": ("vexcel_detect", "vision_inspect"),
    "statistician": ("run_regression",),
    "solar": ("solar_payback", "rank_buildings", "screen_upgrades"),
    "outreach": ("draft_email",),
    "lookup": ("lookup_property",),
    "interior": ("interior_inspect",),
    "writer": ("read_evidence", "list_evidence"),
}
_SUBAGENTS: dict[str, tuple[str, str]] = {
    "sampler": ("Samples buildings for the study city with sample_buildings (call once).",
                prompts.SAMPLER_PROMPT),
    "inspector": ("Inspects ONE building: vexcel_detect then vision_inspect. Give it the "
                  "property id, latitude and longitude.", prompts.INSPECTOR_PROMPT),
    "statistician": ("Runs the regression with run_regression and reports coefficient, 95% CI, "
                     "p-value and n copied from the tool.", prompts.STATISTICIAN_PROMPT),
    "writer": ("Writes the final markdown brief from the evidence ledger, citing every "
               "number with its [E#] tag.", prompts.WRITER_PROMPT),
}
_RETROFIT_SUBAGENTS: dict[str, tuple[str, str]] = {
    "sampler": _SUBAGENTS["sampler"],
    "inspector": (_SUBAGENTS["inspector"][0] + " Reports roof_area_sq_ft.", prompts.INSPECTOR_PROMPT),
    "solar": ("Estimates rooftop solar for every inspected property id (solar_payback each), "
              "then ranks them once (rank_buildings).", prompts.SOLAR_PROMPT),
    "outreach": ("Drafts (never sends) owner emails with draft_email for the top N ranked "
                 "property ids.", prompts.OUTREACH_PROMPT),
    "writer": ("Writes the retrofit brief with a cited ranking table from the evidence ledger.",
               prompts.RETROFIT_WRITER_PROMPT),
}
_PROPERTY_SUBAGENTS: dict[str, tuple[str, str]] = {
    "lookup": ("Looks up ONE CoStar property by id or street address with lookup_property (call once).",
               prompts.LOOKUP_PROMPT),
    "inspector": _RETROFIT_SUBAGENTS["inspector"],
    "interior": ("Optional: checks the property's Matterport scan with interior_inspect; skips cleanly "
                 "when there is none.", prompts.INTERIOR_PROMPT),
    "solar": ("Estimates rooftop solar (solar_payback), ranks it (rank_buildings) and screens all five "
              "upgrades (screen_upgrades) for the property.", prompts.SOLAR_PROMPT),
    "outreach": ("Drafts (never sends) the owner email with draft_email.", prompts.OUTREACH_PROMPT),
    "writer": ("Writes the property report with cited numbers from the evidence ledger.",
               prompts.PROPERTY_WRITER_PROMPT),
}
SUBAGENTS_BY_INTENT = {"research": _SUBAGENTS, "retrofit": _RETROFIT_SUBAGENTS, "property": _PROPERTY_SUBAGENTS}
MOCK_LABEL = "(demo data)"
# Ledger tools every research study must have run successfully, and who runs them.
REQUIRED_STEPS: tuple[tuple[str, str], ...] = (
    ("sample_buildings", "sampler"), ("vision_inspect", "inspector"), ("run_regression", "statistician"),
)
# Retrofit studies need a ranking and at least one owner-email draft instead of a regression.
RETROFIT_REQUIRED_STEPS: tuple[tuple[str, str], ...] = (
    ("sample_buildings", "sampler"), ("vision_inspect", "inspector"),
    ("rank_buildings", "solar"), ("draft_email", "outreach"),
)
# A single-property report needs the property, an inspection and the upgrade screening.
PROPERTY_REQUIRED_STEPS: tuple[tuple[str, str], ...] = (
    ("lookup_property", "lookup"), ("vision_inspect", "inspector"), ("screen_upgrades", "solar"),
)
STEPS_BY_INTENT = {"research": REQUIRED_STEPS, "retrofit": RETROFIT_REQUIRED_STEPS,
                   "property": PROPERTY_REQUIRED_STEPS}


class Agent(Protocol):
    """The part of a compiled LangGraph agent that run_question uses."""

    def invoke(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the agent and return its final state with a "messages" list."""


AgentFactory = Callable[[list[BaseTool]], Agent]


@dataclass
class PlannerRun:
    """A StudyResult plus the run's ledger and citation-check outcome (not schema fields)."""

    study: StudyResult
    ledger: Ledger
    verified: bool
    problems: list[str] = field(default_factory=list)
    handoffs: list[dict[str, str]] = field(default_factory=list)  # {subagent, task, ts}
    planning: list[dict[str, Any]] = field(default_factory=list)  # plan / note events (write_todos, write_file)
    complete: bool = True  # False when a required step (e.g. a regression result) is missing

    def to_json(self) -> dict[str, JsonValue]:
        """Return the study, ledger and verification result as JSON data."""
        return {"study": self.study.model_dump(mode="json"), "ledger": self.ledger.to_json(),
                "verified": self.verified, "complete": self.complete, "problems": list(self.problems)}


def build_agent(tools: Sequence[BaseTool], model: BaseChatModel | str, intent: str = "research") -> Any:
    """Create the orchestrator Deep Agent and the subagents for this intent.

    research: sampler, inspector, statistician, writer. retrofit: sampler, inspector,
    solar, outreach and the retrofit writer.

    Uses deepagents 0.7.38 create_deep_agent(model, tools, system_prompt, subagents,
    name); each declarative SubAgent gets its own prompt and only its own tools. The
    orchestrator itself only gets the read-only evidence tools.
    """
    by_name = {tool.name: tool for tool in tools}
    subagents = [
        {"name": name, "description": description, "system_prompt": prompt,
         "tools": [by_name[t] for t in SUBAGENT_TOOLS[name]]}
        for name, (description, prompt) in SUBAGENTS_BY_INTENT[intent].items()
    ]
    from langchain.agents.middleware import TodoListMiddleware

    # deepagents 0.7.38 does not add a to-do tool by default; TodoListMiddleware gives the
    # orchestrator LangChain's real write_todos tool, which the UI shows as the plan.
    return create_deep_agent(
        model=model,
        tools=[by_name["list_evidence"], by_name["read_evidence"]],
        system_prompt=prompts.ORCHESTRATOR_PROMPT,
        subagents=subagents,
        middleware=[TodoListMiddleware()],
        name="CityLab",
    )


def _default_factory(intent: str) -> AgentFactory:
    """Return a factory building the real Deep Agent with the configured chat model."""
    return lambda tools: build_agent(tools, get_chat_model("chat"), intent)


class _HandOffPrinter(BaseCallbackHandler):
    """Print each subagent hand-off (a `task` tool call) as it starts."""

    raise_error = False

    def __init__(self, handoffs: list[dict[str, str]] | None = None,
                 on_handoff: Callable[[dict[str, str]], None] | None = None,
                 planning: list[dict[str, Any]] | None = None,
                 on_planning: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Collect each hand-off (subagent, redacted short task, ts) in *handoffs* and
        pass it to *on_handoff* as it happens. DeepAgents' own planning tools
        (write_todos, write_file) become plan / note events in *planning*."""
        super().__init__()
        self.handoffs = handoffs if handoffs is not None else []
        self.on_handoff = on_handoff
        self.planning = planning if planning is not None else []
        self.on_planning = on_planning

    def _planning(self, name: str, inputs: dict[str, Any]) -> None:
        """Record a write_todos plan or a write_file note (never file contents)."""
        now = datetime.now(timezone.utc).isoformat()
        if name == "write_todos":
            event = plan_event(list(inputs.get("todos") or []), now)
            print(f"PLAN {event['message']}", flush=True)
        else:
            event = note_event(str(inputs.get("file_path") or inputs.get("path") or ""), len(str(inputs.get("content") or "")), now)
        self.planning.append(event)
        if self.on_planning:
            self.on_planning(event)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *,
                      inputs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Print 'HAND-OFF -> <subagent>: <short task>' for task calls; record plans and notes."""
        name = (serialized or {}).get("name")
        if name not in ("task", "write_todos", "write_file"):
            return
        if inputs is None:
            try:
                inputs = json.loads(input_str)
            except (TypeError, ValueError):
                inputs = {}
        if name != "task":
            self._planning(name, inputs or {})
            return
        target = str((inputs or {}).get("subagent_type", "?"))
        description = " ".join(str((inputs or {}).get("description", "")).split())
        target, task = str(redact(target)), str(redact(description))[:100]
        record = {"subagent": target, "task": task, "ts": datetime.now(timezone.utc).isoformat()}
        self.handoffs.append(record)
        print(f"HAND-OFF -> {target}: {task}", flush=True)
        if self.on_handoff:
            self.on_handoff(record)


def _print_tool_call(entry: dict[str, Any]) -> None:
    """Print one recorded tool call: evidence id, tool name, short args and source."""
    status = " ERROR" if "error" in entry["result"] else ""
    print(f"  TOOL {entry['id']} {entry['tool']} {short_args(entry['args'])} [{entry['source']}]{status}",
          flush=True)


def _text(message: BaseMessage) -> str:
    """Return the plain text of a message."""
    return str(message.text).strip()


def _strip_preamble(text: str) -> str:
    """Drop chatter before the brief's first markdown heading, if it has one."""
    match = re.search(r"^#{1,6}\s", text, re.M)
    return text[match.start():].strip() if match else text


def _lookup_failure_reason(ledger: Ledger) -> str | None:
    """The error message of the latest failed lookup_property call, if any."""
    for entry in reversed(ledger.all()):
        if entry["tool"] == "lookup_property" and "error" in entry["result"]:
            return str(entry["result"]["error"])
    return None


def _property_not_found_brief(question: str, reason: str) -> str:
    """A clean, honest brief for a single-property lookup that never found a building.

    Synthesized here rather than trusted from the model: when lookup_property fails,
    the orchestrator sometimes gives up and writes its own ad-hoc reply (for example
    asking the user to resend the address with a city and state, which a one-shot API
    call has no way to receive) instead of properly delegating to the writer. That
    reply would otherwise become the "brief" verbatim and fail the citation check as
    unverified chatter, so this case is handled directly instead.
    """
    return (f"## Question\n{question}\n\n## Property\n\n**Study Status: Incomplete**\n\n"
            f"{reason} No building record was available, so no inspection, upgrade screening "
            "or outreach draft could be produced.\n\n## Next step\n\nTry a property already in the "
            "sample (a Houston or Los Angeles apartment community), or a CoStar property id.\n")


def _brief_from(messages: list[BaseMessage]) -> str:
    """Prefer the writer subagent's own output; fall back to the final AI message."""
    writer_calls = {call["id"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
                    if call["name"] == "task" and call["args"].get("subagent_type") == "writer"}
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.tool_call_id in writer_calls and _text(message):
            return _strip_preamble(_text(message))
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and _text(message):
            return _strip_preamble(_text(message))
    raise ValueError("The agent did not return a brief.")


def _problems(brief: str, ledger: Ledger) -> list[str]:
    """Run the citation check plus the demo-data label check."""
    problems = verify_brief(brief, ledger)
    if any(e["source"] == "mock" for e in ledger.all()) and MOCK_LABEL not in brief:
        problems.append(f'mock results are not labelled "{MOCK_LABEL}"')
    return problems




_RESULT_CHECKS = {"run_regression": regression_ok, "rank_buildings": ranking_ok, "draft_email": draft_ok}


def _step_done(entry: dict[str, Any]) -> bool:
    """A step counts only if it succeeded and, for some tools, produced a usable result."""
    if "error" in entry["result"]:
        return False
    check = _RESULT_CHECKS.get(entry["tool"])
    return check(entry["result"]) if check else True


def missing_steps(ledger: Ledger, intent: str = "research") -> list[str]:
    """Return the intent's required tools with no successful ledger entry.

    run_regression counts only with the predictor's coefficient and a sample_size;
    rank_buildings only if it ranked a building; draft_email only with an email body.
    """
    done = {e["tool"] for e in ledger.all() if _step_done(e)}
    return [tool for tool, _ in STEPS_BY_INTENT[intent] if tool not in done]


def _incomplete_caveat(step: str) -> str:
    """Return the caveat for one missing required step."""
    return REGRESSION_INCOMPLETE if step == "run_regression" else f"Study incomplete: {step} was not run."


def _study(question: str, brief: str, ledger: Ledger, caveats: list[str], intent: str = "research") -> StudyResult:
    """Assemble a StudyResult exactly per backend/schemas.py from ledger results.

    retrofit: stats is None; solar_results is the latest successful ranking (in rank
    order) and outreach_drafts are the successful email drafts.
    """
    entries = [e for e in ledger.all() if "error" not in e["result"]]
    samples = [e for e in entries if e["tool"] in ("sample_buildings", "lookup_property")]
    if not samples:
        buildings = []
    elif samples[-1]["tool"] == "lookup_property":  # a property report covers the one looked-up building
        buildings = [Building.model_validate(samples[-1]["result"]["building"])]
    else:
        buildings = [Building.model_validate(b) for b in samples[-1]["result"]["buildings"]]
    latest: dict[Any, Measurement] = {}
    for e in entries:
        if e["tool"] == "vision_inspect":
            m = Measurement.model_validate(e["result"])
            latest[m.property_id] = m
    is_mock = any(e["source"] == "mock" for e in ledger.all())
    solar_results: list[SolarResult] | None = None
    outreach_drafts: list[OutreachDraft] | None = None
    if intent in ("retrofit", "property"):
        stats = None
        rankings = [e for e in entries if e["tool"] == "rank_buildings" and ranking_ok(e["result"])]
        solar_results = [SolarResult.model_validate(r) for r in rankings[-1]["result"]["ranked"]] if rankings else []
        outreach_drafts = [OutreachDraft.model_validate(e["result"]) for e in entries
                           if e["tool"] == "draft_email" and draft_ok(e["result"])]
        all_caveats = list(dict.fromkeys(c for r in solar_results for c in (r.caveats or [])))
    else:
        regressions = [e for e in entries if e["tool"] == "run_regression"]
        successful = [e for e in regressions if regression_ok(e["result"])]
        chosen = (successful or regressions)[-1] if regressions else None
        stats = StatsResult.model_validate(chosen["result"]) if chosen else None
        all_caveats = list((stats.caveats if stats else None) or [])
    if is_mock:
        all_caveats.append(f"Results marked {MOCK_LABEL} come from synthetic mock tools, not observations.")
    all_caveats.extend(caveats)
    return StudyResult(id=f"study-{uuid4().hex[:12]}", question=question, buildings=buildings,
                       measurements=list(latest.values()), stats=stats, brief_markdown=brief,
                       caveats=all_caveats, solar_results=solar_results,
                       outreach_drafts=outreach_drafts, is_mock=is_mock)


def _safe_emitter(on_event: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None]:
    """Wrap on_event so a failing listener can never break the run."""
    def emit(event: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # noqa: BLE001 - a listener error must not stop the study
            pass
    return emit


def run_question(question: str, mode: str | None = None, n: int = DEFAULT_N,
                 agent_factory: AgentFactory | None = None, intent: str = "auto",
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> PlannerRun:
    """Run one question through the agent, verify the brief's citations, return a PlannerRun.

    PlannerRun.study is the StudyResult; PlannerRun.ledger and .verified carry the
    evidence and citation-check outcome, which StudyResult has no fields for.
    intent "auto" picks retrofit when the question mentions solar, rooftop, retrofit or
    "best candidates", else research. *on_event*, if given, receives each handoff,
    tool_call (after it is recorded in the ledger) and the final citation_check event
    live, in the shape backend.agent.events builds; without it nothing changes.
    """
    if not question.strip():
        raise ValueError("Provide a nonempty research question.")
    tool_mode = resolve_mode(mode)
    study_intent = detect_intent(question, intent)
    n = max(1, min(int(n), MAX_N))
    top_n = 1 if study_intent == "property" else parse_top_n(question)
    ledger = Ledger()
    emit = _safe_emitter(on_event)

    def on_call(entry: dict[str, Any]) -> None:
        """Print the recorded tool call, then stream it."""
        _print_tool_call(entry)
        emit(tool_call_event(entry))

    tools = get_tools(tool_mode, ledger, n=n, on_call=on_call, question=question,
                      intent=study_intent, top_n=top_n)
    agent = (agent_factory or _default_factory(study_intent))(tools)
    handoffs: list[dict[str, str]] = []
    planning: list[dict[str, Any]] = []
    printer = _HandOffPrinter(handoffs, on_handoff=lambda h: emit(handoff_event(h)), planning=planning, on_planning=emit)
    run_config = {"callbacks": [printer], "recursion_limit": 1000}

    prompt = (f"Question: {question}\nIntent: {study_intent}.\nSample size n: {n} (max {MAX_N}).\n"
              f"Tool mode: {tool_mode}." + (f"\nTop N for outreach drafts: {top_n}." if study_intent in ("retrofit", "property") else ""))
    state = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=run_config)
    messages = list(state.get("messages", []))

    missing = missing_steps(ledger, study_intent)
    if missing:
        owners = dict(STEPS_BY_INTENT[study_intent])
        names = ", ".join(missing)
        who = ", then ".join(owners[m] for m in missing)
        print(f"STEP GUARD: {names} was not run or produced no result; asking the orchestrator once.", flush=True)
        guard = HumanMessage(content=(
            f"The study is incomplete: {names} was not run or did not produce a result. Delegate to "
            f"the {who} now, then to the writer for an updated brief that uses the new evidence. "
            "Return only the brief."))
        state = agent.invoke({"messages": [*messages, guard]}, config=run_config)
        messages = list(state.get("messages", []))
        missing = missing_steps(ledger, study_intent)

    # A single-property lookup that never found a building sometimes makes the
    # orchestrator give up and reply directly (for example asking for a city and state
    # a one-shot call cannot receive) instead of delegating to the writer. Write a
    # clean, honest brief ourselves rather than trust whatever it said.
    lookup_reason = _lookup_failure_reason(ledger) if study_intent == "property" and "lookup_property" in missing else None
    if lookup_reason:
        print("LOOKUP GUARD: lookup_property failed; writing the incomplete brief directly.", flush=True)
        brief = str(redact(_property_not_found_brief(question, lookup_reason)))
        problems: list[str] = []
    else:
        brief = str(redact(_brief_from(messages)))
        problems = _problems(brief, ledger)

    if problems:
        listed = "\n".join(f"- {p}" for p in problems)
        fix = HumanMessage(content=(
            "The automatic citation check failed with these problems:\n" + listed + "\n\n"
            "Delegate to the writer ONCE to fix them. Every number must be copied from the ledger "
            "and immediately followed by the [E#] tag of the entry that contains it; remove any "
            "number that is not in the ledger. Return only the corrected brief."))
        print(f"CITATION CHECK found {len(problems)} problems; asking the writer to fix them.", flush=True)
        state = agent.invoke({"messages": [*messages, fix]}, config=run_config)
        brief = str(redact(_brief_from(list(state.get("messages", [])))))
        problems = _problems(brief, ledger)

    incomplete = [_incomplete_caveat(step) for step in missing]
    caveats = list(incomplete)
    if problems:
        caveat = "Unverified numbers: " + "; ".join(problems)
        caveats.append(caveat)
        brief = brief.rstrip() + "\n\n**Unverified numbers** (failed the automatic citation check):\n" + \
            "\n".join(f"- {p}" for p in problems) + "\n"
    if incomplete:
        brief = brief.rstrip() + "\n\n" + "\n".join(f"**{c}**" for c in incomplete) + "\n"
    study = _study(question, brief, ledger, caveats, study_intent)
    run = PlannerRun(study=study, ledger=ledger, verified=not problems and not missing,
                     problems=problems + incomplete, handoffs=handoffs, complete=not missing, planning=planning)
    emit(citation_event(run.verified, run.complete, run.problems, datetime.now(timezone.utc).isoformat()))
    return run


def plan_research(question: str) -> dict[str, JsonValue]:
    """Retain the scaffold entry point as a JSON wrapper around run_question."""
    return run_question(question).to_json()


def verdict_line(run: PlannerRun) -> str:
    """Return the CLI verdict; an incomplete study always fails, whatever the citations."""
    if not run.complete:
        return "CITATION CHECK: FAIL (study incomplete)"
    return "CITATION CHECK: PASS" if run.verified else f"CITATION CHECK: FAIL ({len(run.problems)} problems)"


def tools_used(ledger: Ledger) -> list[tuple[str, str, int]]:
    """Return (tool, source, call count) rows in first-use order."""
    counts = Counter((e["tool"], e["source"]) for e in ledger.all())
    return [(tool, source, count) for (tool, source), count in counts.items()]


def main(argv: list[str] | None = None) -> int:
    """Run one quoted question from python -m backend.agent.planner."""
    parser = argparse.ArgumentParser(description="Run a CityLab research question.")
    parser.add_argument("question", help="A quoted CityLab research question.")
    parser.add_argument("--mode", choices=MODES, default=None, help="Tool mode (default: TOOLS_MODE or auto).")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Sample size (max {MAX_N}).")
    parser.add_argument("--intent", choices=INTENT_CHOICES, default="auto",
                        help="research (regression) or retrofit (rooftop solar ranking + drafts); "
                             "auto detects retrofit from solar/rooftop/retrofit/'best candidates'.")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Write StudyResult plus ledger here. Use a path outside the repo or under "
                             "data/cache/ (ignored by git), e.g. data/cache/runs/out.json.")
    args = parser.parse_args(argv)
    try:
        run = run_question(args.question, mode=args.mode, n=args.n, intent=args.intent)
    except Exception as exc:
        # SDK/graph exception text can include request details or credentials.
        parser.exit(1, f"CityLab could not complete the question ({type(exc).__name__}); "
                       "check model settings and tool availability.\n")
    print("\n" + run.study.brief_markdown.rstrip() + "\n")
    print(verdict_line(run))
    rows = tools_used(run.ledger)
    width = max([len("tool"), *(len(r[0]) for r in rows)])
    print("TOOLS USED:")
    print(f"  {'tool'.ljust(width)}  source  calls")
    for tool, source, count in rows:
        print(f"  {tool.ljust(width)}  {source.ljust(6)}  {count}")
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run.to_json(), indent=2), encoding="utf-8")
        print(f"Wrote {args.json_path}")
    sys.stdout.flush()
    return 0 if run.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
