"""Project-owned message normalization for agent trajectory training.

These helpers preserve the behavior that the paper's training pipeline used
from its smolagents fork, without tying dataset preprocessing to a particular
smolagents version.
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
import re
from typing import Any


SUPPORTED_ROLES = {"system", "user", "assistant", "tool-call", "tool-response"}
ROLE_ALIASES = {
    "MessageRole.SYSTEM": "system",
    "MessageRole.USER": "user",
    "MessageRole.ASSISTANT": "assistant",
    "MessageRole.TOOL_CALL": "tool-call",
    "MessageRole.TOOL_RESPONSE": "tool-response",
}


def _role_value(role: Any) -> str:
    if isinstance(role, Enum):
        role = role.value
    role = str(role)
    return ROLE_ALIASES.get(role, role)


def normalize_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with legacy enum-style role names normalized."""
    normalized = deepcopy(messages)
    for message in normalized:
        message["role"] = _role_value(message["role"])
    return normalized


def remove_tool_call_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove the redundant explicit tool-call turn from code-agent logs."""
    return [
        deepcopy(message)
        for message in messages
        if _role_value(message["role"]) != "tool-call"
    ]


def get_clean_message_list(
    message_list: list[dict[str, Any]],
    role_conversions: dict[Any, Any] | None = None,
    *,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """Normalize roles and merge adjacent turns with the same role.

    This is the text-only subset used by the math experiments. Images are
    deliberately rejected when flattening instead of being silently lost.
    """
    conversions = {
        _role_value(source): _role_value(target)
        for source, target in (role_conversions or {}).items()
    }
    output: list[dict[str, Any]] = []

    for source_message in deepcopy(message_list):
        role = _role_value(source_message["role"])
        if role not in SUPPORTED_ROLES:
            raise ValueError(
                f"Incorrect role {role!r}; supported roles are {sorted(SUPPORTED_ROLES)}"
            )
        role = conversions.get(role, role)
        content = source_message.get("content", "")

        if flatten_messages_as_text and isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if block.get("type") == "image":
                    raise ValueError("Cannot flatten image content into a text-only SFT example")
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            content = "".join(text_parts)

        if output and output[-1]["role"] == role:
            if flatten_messages_as_text:
                output[-1]["content"] += str(content)
            else:
                previous = output[-1]["content"]
                if isinstance(previous, list) and isinstance(content, list):
                    previous.extend(content)
                else:
                    output[-1]["content"] = f"{previous}{content}"
        else:
            output.append({"role": role, "content": content})

    return output


def prepare_sft_messages(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Convert a raw paper-style trajectory into student SFT messages."""
    normalized = normalize_roles(messages)
    if sum(message["role"] == "system" for message in normalized) != 1:
        raise ValueError("A trajectory must contain exactly one system message")

    normalized = remove_tool_call_from_messages(normalized)
    cleaned = get_clean_message_list(
        normalized,
        role_conversions={"tool-response": "user", "tool-call": "assistant"},
        flatten_messages_as_text=True,
    )

    for message in cleaned:
        if message["role"] == "user":
            message["content"] = re.sub(
                r"<reference>.*?</reference>",
                "",
                message["content"],
                flags=re.DOTALL,
            )
            break

    if cleaned and cleaned[-1]["role"] == "user":
        cleaned = cleaned[:-1]
    for message in cleaned:
        if message["role"] != "assistant":
            continue
        content = message["content"].lstrip()
        if content.startswith("Thought:"):
            continue
        # A small number of otherwise valid teacher turns place a short
        # conclusion before the Thought marker, or omit only the marker. Keep
        # every word while normalizing the action prefix expected at inference.
        marker = "\n\nThought:"
        if marker in content:
            preamble, reasoning = content.split(marker, 1)
            content = f"Thought: {preamble.strip()}\n\n{reasoning.lstrip()}"
        else:
            content = f"Thought: {content}"
        message["content"] = content
    if system_prompt is not None:
        cleaned[0]["content"] = system_prompt
    return cleaned


__all__ = [
    "get_clean_message_list",
    "normalize_roles",
    "prepare_sft_messages",
    "remove_tool_call_from_messages",
]
