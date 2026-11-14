"""Test provider selection and multimodal messages without any live model calls."""

import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from backend import llm


@pytest.mark.parametrize(
    ("provider", "module", "class_name", "extra"),
    [
        ("bedrock", "langchain_aws", "ChatBedrockConverse", {"region_name": "us-east-1"}),
        ("google", "langchain_google_genai", "ChatGoogleGenerativeAI", {"api_key": "test-google-sentinel"}),
        ("anthropic", "langchain_anthropic", "ChatAnthropic", {}),
    ],
)
def test_get_chat_model_selects_configured_provider(
    provider: str, module: str, class_name: str, extra: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the configured model, region and Google credential to the right adapter."""
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{class_name: constructor}))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", f" {provider.upper()} ")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "test-vision-model")
    monkeypatch.setattr(llm.config, "AWS_REGION", "us-east-1")
    monkeypatch.setattr(llm.config, "GOOGLE_API_KEY", "test-google-sentinel")
    assert llm.get_chat_model() is model
    constructor.assert_called_once_with(model="test-vision-model", **extra)


@pytest.mark.parametrize(
    ("provider", "model", "message"),
    [(None, "model", "LLM_PROVIDER"), ("unknown", "model", "LLM_PROVIDER"),
     ("bedrock", "  ", "LLM_MODEL")],
)
def test_invalid_configuration_fails_before_provider_creation(
    provider: str | None, model: str, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or unsupported settings produce clear, credential-free errors."""
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm.config, "LLM_MODEL", model)
    with pytest.raises(ValueError, match=message):
        llm.get_chat_model()


def test_google_requires_its_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently switch authentication methods when the requested key is absent."""
    constructor = Mock()
    monkeypatch.setitem(sys.modules, "langchain_google_genai", SimpleNamespace(ChatGoogleGenerativeAI=constructor))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "google")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "test-model")
    monkeypatch.setattr(llm.config, "GOOGLE_API_KEY", None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        llm.get_chat_model()
    constructor.assert_not_called()


def test_chat_and_vision_roles_choose_separate_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep no-argument chat calls compatible while routing images to VISION_MODEL."""
    constructor = Mock()
    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrockConverse=constructor))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "chat-model")
    monkeypatch.setattr(llm.config, "VISION_MODEL", "vision-model")
    monkeypatch.setattr(llm.config, "AWS_REGION", "us-east-1")
    llm.get_chat_model()
    constructor.assert_called_with(model="chat-model", region_name="us-east-1")
    llm.get_chat_model("vision")
    constructor.assert_called_with(model="vision-model", region_name="us-east-1")
    monkeypatch.setattr(llm.config, "VISION_MODEL", None)
    with pytest.raises(ValueError, match="VISION_MODEL"):
        llm.get_chat_model("vision")
    with pytest.raises(ValueError, match="role"):
        llm.get_chat_model("unknown")


@pytest.mark.parametrize("role", ["chat", "vision"])
def test_explicit_model_override_works_with_both_roles(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner's role API and the AI tools' model-id override coexist."""
    constructor = Mock()
    monkeypatch.setitem(sys.modules, "langchain_aws", SimpleNamespace(ChatBedrockConverse=constructor))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(llm.config, "LLM_MODEL", None)
    monkeypatch.setattr(llm.config, "VISION_MODEL", None)
    monkeypatch.setattr(llm.config, "AWS_REGION", "us-east-1")
    llm.get_chat_model(role, model_id="override-model")
    constructor.assert_called_once_with(model="override-model", region_name="us-east-1")


def test_ai_text_and_image_helpers_forward_model_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision and outreach helpers retain explicit model overrides after the merge."""
    path = tmp_path / "aerial.jpg"
    Image.new("RGB", (2, 2), "green").save(path, format="JPEG")
    model = Mock()
    model.invoke.return_value = AIMessage(content="mock reply")
    factory = Mock(return_value=model)
    monkeypatch.setattr(llm, "get_chat_model", factory)
    assert llm.ask_text("Draft an email", model_id="text-override") == "mock reply"
    factory.assert_called_once_with(model_id="text-override")
    factory.reset_mock()
    assert llm.ask_with_image("Measure the roof", str(path), "vision-override") == "mock reply"
    factory.assert_called_once_with("vision", model_id="vision-override")


@pytest.mark.parametrize(
    "content",
    ["An apartment building is surrounded by trees.",
     [{"type": "thinking", "thinking": "private reasoning"},
      {"type": "text", "text": "An apartment "}, {"type": "text", "text": "building."}]],
)
def test_ask_with_image_encodes_jpeg_and_returns_only_reply_text(
    content: str | list[dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the exact local bytes and prompt in one message; omit reasoning blocks."""
    path = tmp_path / "aerial.jpg"
    Image.new("RGB", (2, 2), "green").save(path, format="JPEG")
    model = Mock()
    reply = AIMessage(content=content)
    model.invoke.return_value = reply
    factory = Mock(return_value=model)
    monkeypatch.setattr(llm, "get_chat_model", factory)
    result = llm.ask_with_image("Describe the rooftop.", str(path))
    assert result == str(reply.text)
    assert "private reasoning" not in result
    factory.assert_called_once_with("vision")
    model.invoke.assert_called_once()
    messages = model.invoke.call_args.args[0]
    assert len(messages) == 1 and isinstance(messages[0], HumanMessage)
    text, image = messages[0].content
    assert text == {"type": "text", "text": "Describe the rooftop."}
    assert image["type"] == "image" and image["mime_type"] == "image/jpeg"
    assert base64.b64decode(image["base64"]) == path.read_bytes()


def test_invalid_image_does_not_create_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject mislabeled data before model initialization or a paid invocation."""
    path = tmp_path / "aerial.jpg"
    path.write_text("not a JPEG")
    factory = Mock()
    monkeypatch.setattr(llm, "get_chat_model", factory)
    with pytest.raises(ValueError, match="JPEG"):
        llm.ask_with_image("Describe it.", str(path))
    factory.assert_not_called()


def test_command_uses_requested_prompt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints the mocked answer to the exact one-sentence prompt."""
    ask = Mock(return_value="A flat-roofed building sits beside a parking lot.")
    monkeypatch.setattr(llm, "ask_with_image", ask)
    assert llm.main(["aerial.jpg"]) == 0
    ask.assert_called_once_with("Describe this aerial image in one sentence.", "aerial.jpg")
    assert capsys.readouterr().out.strip() == ask.return_value


def test_command_does_not_echo_provider_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """An SDK exception containing sensitive request details stays out of output."""
    monkeypatch.setattr(llm, "ask_with_image", Mock(side_effect=RuntimeError("private-test-sentinel")))
    with pytest.raises(SystemExit) as error:
        llm.main(["aerial.jpg"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "private-test-sentinel" not in output.out + output.err
    assert "Unable to describe the image" in output.err
