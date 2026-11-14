"""Build and run the CityLab Deep Agent with live or synthetic tool adapters."""

import argparse
import importlib
import json
import os
import re
from collections.abc import Callable
from functools import wraps

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, create_schema_from_function
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, JsonValue

from backend import config
from backend.agent.prompts import SYSTEM_PROMPT
from backend.llm import get_chat_model

# Public agent aliases preserve the canonical functions/signatures in CONTRACTS.md.
_TOOL_BINDINGS = (
    ("costar_data", "load_properties", "load_properties"),
    ("costar_data", "get_property", "get_property"),
    ("costar_data", "find_buildings", "sample_buildings"),
    ("vexcel", "get_aerial", "get_aerial"),
    ("vexcel", "get_property_info", "vexcel_detect"),
    ("vision", "measure_building", "vision_inspect"),
    ("stats", "run_study", "run_regression"),
    ("solar", "estimate_solar", "estimate_solar"),
    ("solar", "rank_buildings", "rank_buildings"),
    ("outreach", "draft_outreach", "draft_outreach"),
    ("matterport", "get_model", "get_model"),
)


def _json_data(value: object) -> JsonValue:
    """Convert schema objects to JSON data at the LangChain tool boundary."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_data(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("A CityLab tool returned a non-JSON value.")


def _redact(value: JsonValue) -> JsonValue:
    """Remove credential values and sensitive argument fields from console output."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if re.search(r"key|secret|token|password|authorization", key, re.I)
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        secrets = [item for name, item in os.environ.items()
                   if item and re.search(r"key|secret|token|password", name, re.I)]
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
    return value


def _wrap_tool(function: Callable[..., object], name: str) -> BaseTool:
    """Retain typed inputs while serializing model returns for the agent."""
    @wraps(function)
    def invoke(*args: object, **kwargs: object) -> JsonValue:
        """Call the selected implementation and expose JSON-serializable evidence."""
        return _json_data(function(*args, **kwargs))

    return StructuredTool.from_function(
        func=invoke, name=name, description=function.__doc__ or name,
        args_schema=create_schema_from_function(name, function),
    )


def _build_tools(use_mock: bool) -> list[BaseTool]:
    """Choose implementations for every contract while keeping stable agent names."""
    tools = []
    for module_name, function_name, public_name in _TOOL_BINDINGS:
        module = importlib.import_module(f"backend.tools.{'mock' if use_mock else module_name}")
        tools.append(_wrap_tool(getattr(module, function_name), public_name))
    return tools


class _ToolCallPrinter(BaseCallbackHandler):
    """Print each actual tool invocation before its body runs, with redaction."""

    raise_error = True

    def on_tool_start(
        self, serialized: dict[str, object], input_str: str, *,
        inputs: dict[str, object] | None = None, **kwargs: object,
    ) -> None:
        """Emit the tool name and its arguments immediately, never credentials."""
        if inputs is None:
            try:
                arguments = json.loads(input_str)
            except (ValueError, TypeError):
                arguments = input_str
        else:
            arguments = inputs
        name = str(serialized.get("name", "tool"))
        encoded = json.dumps(_redact(_json_data(arguments)), ensure_ascii=False, allow_nan=False)
        print(f"{_redact(name)} {encoded}", flush=True)


def build_agent() -> CompiledStateGraph:
    """Build with Deep Agents 0.7.38's inspected model/tools/system_prompt keywords."""
    mode = "MOCK: all CityLab tool numbers are synthetic." if config.USE_MOCK_TOOLS else (
        "REAL tools selected; report unavailable implementations or data instead of inventing outputs."
    )
    return create_deep_agent(
        model=get_chat_model("chat"),
        tools=_build_tools(config.USE_MOCK_TOOLS),
        system_prompt=f"{SYSTEM_PROMPT}\n\nRun mode: {mode}",
        name="CityLab",
    )


def run_question(question: str) -> str:
    """Run a question, print tool names/arguments as they execute, then the final text."""
    if not question.strip():
        raise ValueError("Provide a nonempty research or retrofit question.")
    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": [_ToolCallPrinter()], "recursion_limit": 1000, "max_concurrency": 1},
    )
    messages = result.get("messages", [])
    if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
        raise ValueError("The agent did not return a final answer.")
    answer = str(_redact(str(messages[-1].text))).strip()
    if not answer:
        raise ValueError("The agent returned no final text.")
    if config.USE_MOCK_TOOLS:
        answer = "MOCK / SYNTHETIC DATA\n" + answer
    print(answer, flush=True)
    return answer


def plan_research(question: str) -> dict[str, JsonValue]:
    """Retain the scaffold entry point as a JSON wrapper around run_question."""
    return {"question": str(_redact(question)), "answer": run_question(question)}


def main(argv: list[str] | None = None) -> int:
    """Run one quoted question from python -m backend.agent.planner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="A quoted CityLab research or retrofit question.")
    args = parser.parse_args(argv)
    try:
        run_question(args.question)
    except Exception:
        # SDK/graph exception text can include request details or credentials.
        parser.exit(1, "CityLab could not complete the question; check model settings and tool availability.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
