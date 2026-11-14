"""Choose a LangChain provider and ask questions about local JPEG images."""

import argparse
import base64
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend import config


BEDROCK_MAX_ATTEMPTS = 10


def get_chat_model(
    role: Literal["chat", "vision"] = "chat", *, model_id: str | None = None,
) -> BaseChatModel:
    """Choose a chat/vision model, optionally overriding its id for an AI tool."""
    if role not in {"chat", "vision"}:
        raise ValueError("Model role must be chat or vision.")
    provider = (config.LLM_PROVIDER or "").strip().lower()
    setting = "LLM_MODEL" if role == "chat" else "VISION_MODEL"
    configured = config.LLM_MODEL if role == "chat" else config.VISION_MODEL
    model = ((model_id if model_id is not None else configured) or "").strip()
    if provider not in {"bedrock", "google", "anthropic"}:
        raise ValueError("Set LLM_PROVIDER to bedrock, google, or anthropic in .env.")
    if not model:
        raise ValueError(f"Set {setting} in .env.")

    # Load only the selected provider; importing this module makes no model calls.
    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse

        # Parallel subagents can exceed Bedrock's request rate; botocore's default of 2
        # retries gives up on ThrottlingException, so back off adaptively instead.
        retries = Config(retries={"max_attempts": BEDROCK_MAX_ATTEMPTS, "mode": "adaptive"})
        return ChatBedrockConverse(model=model, region_name=config.AWS_REGION, config=retries)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.GOOGLE_API_KEY:
            raise ValueError("Set GOOGLE_API_KEY in .env for the google provider.")
        return ChatGoogleGenerativeAI(model=model, api_key=config.GOOGLE_API_KEY)

    from langchain_anthropic import ChatAnthropic

    # ChatAnthropic reads ANTHROPIC_API_KEY, loaded into the environment by config.
    return ChatAnthropic(model=model)


def vision_model_id() -> str | None:
    """Return the configured vision override, or None when it is absent."""
    return (config.VISION_MODEL or "").strip() or None


def ask_text(prompt: str, model_id: str | None = None) -> str:
    """Send one text prompt and return only the reply text."""
    model = get_chat_model(model_id=model_id) if model_id else get_chat_model()
    return str(model.invoke([HumanMessage(content=prompt)]).text)


def ask_with_image(prompt: str, image_path: str, model_id: str | None = None) -> str:
    """Send text plus a base64 JPEG in a standard multimodal HumanMessage."""
    image_bytes = Path(image_path).read_bytes()
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise ValueError("image_path must refer to a JPEG image.")
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "base64": base64.b64encode(image_bytes).decode("ascii"),
            "mime_type": "image/jpeg",
        },
    ])
    model = get_chat_model("vision", model_id=model_id) if model_id else get_chat_model("vision")
    reply = model.invoke([message])
    return str(reply.text)


def ask_with_images(prompt: str, image_paths: list[str], model_id: str | None = None) -> str:
    """Send text plus several JPEGs (in order) in one vision message; return only reply text."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        image_bytes = Path(path).read_bytes()
        if not image_bytes.startswith(b"\xff\xd8\xff"):
            raise ValueError("image_paths must refer to JPEG images.")
        content.append({"type": "image", "base64": base64.b64encode(image_bytes).decode("ascii"),
                        "mime_type": "image/jpeg"})
    model = get_chat_model("vision", model_id=model_id) if model_id else get_chat_model("vision")
    return str(model.invoke([HumanMessage(content=content)]).text)


def main(argv: list[str] | None = None) -> int:
    """Describe one aerial JPEG from the command line without logging credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to a local aerial JPEG image.")
    args = parser.parse_args(argv)
    try:
        answer = ask_with_image("Describe this aerial image in one sentence.", args.image_path)
    except Exception:
        # Provider exception bodies can contain request details; do not echo them.
        parser.exit(1, "Unable to describe the image; check the JPEG path, LLM settings, and provider credentials.\n")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
