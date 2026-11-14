"""Exercise the actual Deep Agents graph with scripted models and offline tools."""

import inspect
import json
import runpy
import socket
from collections.abc import Sequence
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, TypeAdapter

from backend.agent import planner
from backend.schemas import Building, SolarResult
from backend.tools import mock


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Use scripted AIMessage tool calls while the real graph executes each tool."""

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "ToolCallingFakeModel":
        """Accept the harness tools without constructing a provider client."""
        return self


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable tracing and fail every attempted network connection in these tests."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(planner.config, "USE_MOCK_TOOLS", True)

    def forbidden(*args: object, **kwargs: object) -> None:
        """Network access is never part of a fake-model test."""
        raise AssertionError("Network access attempted in an offline planner test")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def fake_model(monkeypatch: pytest.MonkeyPatch, responses: list[AIMessage]) -> Mock:
    """Inject a local scripted model with a tokenizer that cannot download files."""
    model = ToolCallingFakeModel(
        responses=responses, custom_get_token_ids=lambda text: [0] * max(1, len(text) // 4),
        profile={"max_input_tokens": 100000, "max_output_tokens": 1024},
    )
    factory = Mock(return_value=model)
    monkeypatch.setattr(planner, "get_chat_model", factory)
    return factory


def call(name: str, args: dict[str, object], index: int) -> AIMessage:
    """Create one deterministic tool request from the fake chat model."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{index}"}])


def test_every_mock_matches_a_contract_and_returns_schema_objects() -> None:
    """Cover all eleven mock functions, including optional Matterport, without keys."""
    building = mock.find_buildings(14, 11, 1)[0]
    image = mock.get_aerial(building.latitude, building.longitude, building.id)
    measurement = mock.measure_building(image.image_path, building.id)
    solar = mock.estimate_solar(building, measurement)
    arguments = {
        "load_properties": {}, "get_property": {"id": building.id},
        "find_buildings": {"market_id": 14, "property_type": 11, "limit": 2},
        "get_aerial": {"lat": building.latitude, "lon": building.longitude, "property_id": building.id},
        "get_property_info": {"lat": building.latitude, "lon": building.longitude},
        "measure_building": {"image_path": image.image_path, "property_id": building.id},
        "run_study": {"question": "green cover and rent", "buildings": [building], "measurements": [measurement]},
        "estimate_solar": {"building": building, "vision_result": measurement},
        "rank_buildings": {"estimates": [solar]},
        "draft_outreach": {"building": building, "solar_estimate": solar},
        "get_model": {"model_id": "mock-model"},
    }
    for module_name, function_name, _ in planner._TOOL_BINDINGS:
        real_module = __import__(f"backend.tools.{module_name}", fromlist=[function_name])
        real = getattr(real_module, function_name)
        function = getattr(mock, function_name)
        assert inspect.signature(function, eval_str=True) == inspect.signature(real, eval_str=True), function_name
        result = function(**arguments[function_name])
        objects = result if isinstance(result, list) else [result]
        assert all(isinstance(item, BaseModel) for item in objects), function_name
        adapter = TypeAdapter(inspect.signature(function).return_annotation)
        assert adapter.validate_json(adapter.dump_json(result)) == result
    assert mock.get_property(-1) is None


def test_mock_cohorts_are_deterministic_and_do_not_share_mutable_models() -> None:
    """Separate city ids/locations and fresh objects prevent fake-data contamination."""
    houston = mock.find_buildings(14, 11, 50)
    la = mock.find_buildings(20, 11, 50)
    assert len(houston) == len(la) == 12
    assert {b.id for b in houston}.isdisjoint(b.id for b in la)
    assert all(b.market_id == 20 and 33 < b.latitude < 35 and -119 < b.longitude < -117 for b in la)
    original_name = houston[0].name
    houston[0].name = "changed by caller"
    assert mock.find_buildings(14, 11, 1)[0].name == original_name
    assert mock.find_buildings(14, 5, 10) == []


@pytest.mark.parametrize("setting, expected", [(None, True), ("true", True), ("false", False), (" TRUE ", True)])
def test_mock_mode_setting(
    setting: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment can explicitly select either tool mode; mocks are the default."""
    # Isolate dotenv loading so the user's private file cannot affect this check.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    if setting is None:
        monkeypatch.delenv("USE_MOCK_TOOLS", raising=False)
    else:
        monkeypatch.setenv("USE_MOCK_TOOLS", setting)
    settings = runpy.run_path(planner.config.__file__)
    assert settings["USE_MOCK_TOOLS"] is expected


def test_solar_rankings_use_payback_and_preserve_inputs() -> None:
    """Missing payback ranks last, with one-based ranks and no email being sent."""
    inputs = [SolarResult(id="unknown", property_id=1),
              SolarResult(id="slow", property_id=2, payback_years=9),
              SolarResult(id="fast", property_id=3, payback_years=6)]
    ranked = mock.rank_buildings(inputs)
    assert [result.id for result in ranked] == ["fast", "slow", "unknown"]
    assert [result.rank for result in ranked] == [1, 2, 3]
    assert all(result.rank is None for result in inputs)


@pytest.mark.parametrize("use_mock", [True, False])
def test_build_agent_selects_the_correct_tool_implementations(
    use_mock: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create_deep_agent receives the chat role, correct tools and the CityLab prompt."""
    monkeypatch.setattr(planner.config, "USE_MOCK_TOOLS", use_mock)
    factory = fake_model(monkeypatch, [AIMessage(content="done")])
    created = object()
    creator = Mock(return_value=created)
    monkeypatch.setattr(planner, "create_deep_agent", creator)
    assert planner.build_agent() is created
    factory.assert_called_once_with("chat")
    kwargs = creator.call_args.kwargs
    assert kwargs["model"] is factory.return_value
    assert kwargs["name"] == "CityLab"
    assert "Never invent numbers" in kwargs["system_prompt"]
    assert {tool.name for tool in kwargs["tools"]} == {binding[2] for binding in planner._TOOL_BINDINGS}
    for tool in kwargs["tools"]:
        function = inspect.unwrap(tool.func)
        assert (function.__module__ == "backend.tools.mock") is use_mock
        tool.args_schema.model_json_schema()


def test_real_tool_mode_does_not_silently_fall_back_to_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real data-source failure remains explicit even after real tools are implemented."""
    from backend.tools import costar_data

    def unavailable(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
        """Represent an unavailable real data source without reading the local kit."""
        raise RuntimeError("real data unavailable")

    monkeypatch.setattr(costar_data, "find_buildings", unavailable)
    tools = {tool.name: tool for tool in planner._build_tools(False)}
    with pytest.raises(RuntimeError, match="real data unavailable"):
        tools["sample_buildings"].invoke({"market_id": 14, "property_type": 11, "limit": 1})


def test_research_runs_real_graph_and_prints_tools_before_the_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Sample, fetch both evidence sources, inspect and regress using a fake model."""
    building = mock.find_buildings(14, 11, 1)[0]
    image = mock.get_aerial(building.latitude, building.longitude, building.id)
    measurement = mock.measure_building(image.image_path, building.id)
    question = "Does green cover correlate with Houston apartment rent?"
    final = "Mock run_regression: effect 18, interval [-4.1, 40.1], sample_size 1; fixed fake statistics, no fit."
    responses = [
        call("sample_buildings", {"market_id": 14, "property_type": 11, "limit": 1}, 1),
        call("get_aerial", {"lat": building.latitude, "lon": building.longitude, "property_id": building.id}, 2),
        call("vexcel_detect", {"lat": building.latitude, "lon": building.longitude}, 3),
        call("vision_inspect", {"image_path": image.image_path, "property_id": building.id}, 4),
        call("run_regression", {"question": question, "buildings": [building.model_dump(mode="json")],
                               "measurements": [measurement.model_dump(mode="json")]}, 5),
        AIMessage(content=final),
    ]
    factory = fake_model(monkeypatch, responses)
    answer = planner.run_question(question)
    output = capsys.readouterr().out
    lines = output.splitlines()
    assert [line.split(" ", 1)[0] for line in lines[:5]] == [
        "sample_buildings", "get_aerial", "vexcel_detect", "vision_inspect", "run_regression",
    ]
    assert json.loads(lines[0].split(" ", 1)[1])["limit"] == 1
    assert answer == "MOCK / SYNTHETIC DATA\n" + final
    assert output.endswith(answer + "\n")
    factory.assert_called_once_with("chat")


def test_retrofit_runs_solar_ranking_and_drafts_owner_email(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A real graph routes the fake solar workflow without regression or sending mail."""
    building = mock.find_buildings(20, 11, 1)[0]
    image = mock.get_aerial(building.latitude, building.longitude, building.id)
    measurement = mock.measure_building(image.image_path, building.id)
    solar = mock.estimate_solar(building, measurement)
    responses = [
        call("sample_buildings", {"market_id": 20, "property_type": 11, "limit": 1}, 1),
        call("get_aerial", {"lat": building.latitude, "lon": building.longitude, "property_id": building.id}, 2),
        call("vexcel_detect", {"lat": building.latitude, "lon": building.longitude}, 3),
        call("vision_inspect", {"image_path": image.image_path, "property_id": building.id}, 4),
        call("estimate_solar", {"building": building.model_dump(mode="json"),
                                "vision_result": measurement.model_dump(mode="json")}, 5),
        call("rank_buildings", {"estimates": [solar.model_dump(mode="json")]}, 6),
        call("draft_outreach", {"building": building.model_dump(mode="json"),
                               "solar_estimate": solar.model_dump(mode="json")}, 7),
        AIMessage(content=f"MOCK: payback {solar.payback_years} years from estimate_solar. Owner email is a draft only."),
    ]
    fake_model(monkeypatch, responses)
    answer = planner.run_question("Rank a Los Angeles apartment for rooftop solar and draft an owner email.")
    output = capsys.readouterr().out
    assert "rank_buildings {" in output and "draft_outreach {" in output
    assert "run_regression {" not in output
    assert "draft only" in answer


def test_console_redacts_sensitive_fields_values_and_final_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Even an unexpected model echo or a credential-shaped argument is redacted."""
    monkeypatch.setenv("VEXCEL_API_KEY", "private-test-value")
    planner._ToolCallPrinter().on_tool_start(
        {"name": "example"}, "", inputs={"api_key": "unrecognized-sensitive-value", "note": "private-test-value"},
    )
    fake_model(monkeypatch, [AIMessage(content="Echo private-test-value")])
    answer = planner.run_question("A harmless question")
    output = capsys.readouterr().out
    assert "private-test-value" not in output + answer
    assert "unrecognized-sensitive-value" not in output
    assert "[REDACTED]" in output


def test_cli_and_empty_question(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI forwards a quoted question and never exposes provider exception details."""
    run = Mock(return_value="answer")
    monkeypatch.setattr(planner, "run_question", run)
    assert planner.main(["Describe the sample"]) == 0
    run.assert_called_once_with("Describe the sample")
    run.side_effect = RuntimeError("private-sdk-detail")
    with pytest.raises(SystemExit) as error:
        planner.main(["Describe the sample"])
    assert error.value.code == 1
    assert "private-sdk-detail" not in capsys.readouterr().err


def test_empty_question_does_not_create_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank question fails before any model/provider initialization."""
    builder = Mock()
    monkeypatch.setattr(planner, "build_agent", builder)
    with pytest.raises(ValueError, match="nonempty"):
        planner.run_question("  ")
    builder.assert_not_called()
