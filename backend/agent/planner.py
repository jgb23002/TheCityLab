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
from backend.agent.evidence import Ledger
from backend.agent.registry import (
    DEFAULT_N, MAX_N, MODES, get_tools, redact, regression_ok, resolve_mode, short_args,
)
from backend.llm import get_chat_model
from backend.schemas import Building, Measurement, StatsResult, StudyResult

SUBAGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "sampler": ("sample_buildings",),
    "inspector": ("vexcel_detect", "vision_inspect"),
    "statistician": ("run_regression",),
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
MOCK_LABEL = "(demo data)"
# Ledger tools every research study must have run successfully, and who runs them.
REQUIRED_STEPS: tuple[tuple[str, str], ...] = (
    ("sample_buildings", "sampler"), ("vision_inspect", "inspector"), ("run_regression", "statistician"),
)


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
    complete: bool = True  # False when a required step (e.g. a regression result) is missing

    def to_json(self) -> dict[str, JsonValue]:
        """Return the study, ledger and verification result as JSON data."""
        return {"study": self.study.model_dump(mode="json"), "ledger": self.ledger.to_json(),
                "verified": self.verified, "complete": self.complete, "problems": list(self.problems)}


def build_agent(tools: Sequence[BaseTool], model: BaseChatModel | str) -> Any:
    """Create the orchestrator Deep Agent with sampler, inspector, statistician and writer.

    Uses deepagents 0.7.38 create_deep_agent(model, tools, system_prompt, subagents,
    name); each declarative SubAgent gets its own prompt and only its own tools. The
    orchestrator itself only gets the read-only evidence tools.
    """
    by_name = {tool.name: tool for tool in tools}
    subagents = [
        {"name": name, "description": description, "system_prompt": prompt,
         "tools": [by_name[t] for t in SUBAGENT_TOOLS[name]]}
        for name, (description, prompt) in _SUBAGENTS.items()
    ]
    return create_deep_agent(
        model=model,
        tools=[by_name["list_evidence"], by_name["read_evidence"]],
        system_prompt=prompts.ORCHESTRATOR_PROMPT,
        subagents=subagents,
        name="CityLab",
    )


def _default_factory(tools: list[BaseTool]) -> Agent:
    """Build the real Deep Agent with the configured chat model from backend/llm.py."""
    return build_agent(tools, get_chat_model("chat"))


class _HandOffPrinter(BaseCallbackHandler):
    """Print each subagent hand-off (a `task` tool call) as it starts."""

    raise_error = False

    def __init__(self, handoffs: list[dict[str, str]] | None = None) -> None:
        """Optionally collect each hand-off (subagent, redacted short task, ts) in *handoffs*."""
        super().__init__()
        self.handoffs = handoffs if handoffs is not None else []

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *,
                      inputs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Print 'HAND-OFF -> <subagent>: <short task>' for task tool calls only."""
        if (serialized or {}).get("name") != "task":
            return
        if inputs is None:
            try:
                inputs = json.loads(input_str)
            except (TypeError, ValueError):
                inputs = {}
        target = str((inputs or {}).get("subagent_type", "?"))
        description = " ".join(str((inputs or {}).get("description", "")).split())
        target, task = str(redact(target)), str(redact(description))[:100]
        self.handoffs.append({"subagent": target, "task": task, "ts": datetime.now(timezone.utc).isoformat()})
        print(f"HAND-OFF -> {target}: {task}", flush=True)


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


REGRESSION_INCOMPLETE = "Study incomplete: regression did not produce a result"


def _step_done(entry: dict[str, Any]) -> bool:
    """A step counts only if it succeeded; a regression must also have produced a result."""
    if "error" in entry["result"]:
        return False
    return regression_ok(entry["result"]) if entry["tool"] == "run_regression" else True


def missing_steps(ledger: Ledger) -> list[str]:
    """Return REQUIRED_STEPS tools with no successful ledger entry.

    run_regression counts only when its result contains the predictor's coefficient
    and a sample_size (a StatsResult with findings.error is not a result).
    """
    done = {e["tool"] for e in ledger.all() if _step_done(e)}
    return [tool for tool, _ in REQUIRED_STEPS if tool not in done]


def _incomplete_caveat(step: str) -> str:
    """Return the caveat for one missing required step."""
    return REGRESSION_INCOMPLETE if step == "run_regression" else f"Study incomplete: {step} was not run."


def _study(question: str, brief: str, ledger: Ledger, caveats: list[str]) -> StudyResult:
    """Assemble a StudyResult exactly per backend/schemas.py from ledger results."""
    entries = [e for e in ledger.all() if "error" not in e["result"]]
    samples = [e for e in entries if e["tool"] == "sample_buildings"]
    buildings = [Building.model_validate(b) for b in samples[-1]["result"]["buildings"]] if samples else []
    latest: dict[Any, Measurement] = {}
    for e in entries:
        if e["tool"] == "vision_inspect":
            m = Measurement.model_validate(e["result"])
            latest[m.property_id] = m
    regressions = [e for e in entries if e["tool"] == "run_regression"]
    successful = [e for e in regressions if regression_ok(e["result"])]
    chosen = (successful or regressions)[-1] if regressions else None
    stats = StatsResult.model_validate(chosen["result"]) if chosen else None
    is_mock = any(e["source"] == "mock" for e in ledger.all())
    all_caveats = list((stats.caveats if stats else None) or [])
    if is_mock:
        all_caveats.append(f"Results marked {MOCK_LABEL} come from synthetic mock tools, not observations.")
    all_caveats.extend(caveats)
    return StudyResult(id=f"study-{uuid4().hex[:12]}", question=question, buildings=buildings,
                       measurements=list(latest.values()), stats=stats, brief_markdown=brief,
                       caveats=all_caveats, is_mock=is_mock)


def run_question(question: str, mode: str | None = None, n: int = DEFAULT_N,
                 agent_factory: AgentFactory | None = None) -> PlannerRun:
    """Run one question through the agent, verify the brief's citations, return a PlannerRun.

    PlannerRun.study is the StudyResult; PlannerRun.ledger and .verified carry the
    evidence and citation-check outcome, which StudyResult has no fields for.
    """
    if not question.strip():
        raise ValueError("Provide a nonempty research question.")
    tool_mode = resolve_mode(mode)
    n = max(1, min(int(n), MAX_N))
    ledger = Ledger()
    tools = get_tools(tool_mode, ledger, n=n, on_call=_print_tool_call, question=question)
    agent = (agent_factory or _default_factory)(tools)
    handoffs: list[dict[str, str]] = []
    run_config = {"callbacks": [_HandOffPrinter(handoffs)], "recursion_limit": 1000}

    prompt = (f"Question: {question}\nSample size n: {n} (max {MAX_N}).\nTool mode: {tool_mode}.")
    state = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=run_config)
    messages = list(state.get("messages", []))

    missing = missing_steps(ledger)
    if missing:
        owners = dict(REQUIRED_STEPS)
        names = ", ".join(missing)
        who = ", then ".join(owners[m] for m in missing)
        print(f"STEP GUARD: {names} was not run or produced no result; asking the orchestrator once.", flush=True)
        guard = HumanMessage(content=(
            f"The study is incomplete: {names} was not run or did not produce a result. Delegate to "
            f"the {who} now, then to the writer for an updated brief that uses the new evidence. "
            "Return only the brief."))
        state = agent.invoke({"messages": [*messages, guard]}, config=run_config)
        messages = list(state.get("messages", []))
        missing = missing_steps(ledger)

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
    study = _study(question, brief, ledger, caveats)
    return PlannerRun(study=study, ledger=ledger, verified=not problems and not missing,
                      problems=problems + incomplete, handoffs=handoffs, complete=not missing)


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
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Write StudyResult plus ledger here. Use a path outside the repo or under "
                             "data/cache/ (ignored by git), e.g. data/cache/runs/out.json.")
    args = parser.parse_args(argv)
    try:
        run = run_question(args.question, mode=args.mode, n=args.n)
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
