"""Message construction for one-step teacher repair and SFT output."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from smolagents.models import ChatMessage

from exps_research.train_utils.message_utils import prepare_sft_messages


REPAIR_PROMPT_VERSION = "trajectory-prefix-only-v1"


def repair_prompt_sha256() -> str:
    """Hash the intentionally empty additional repair instruction."""
    return hashlib.sha256(b"").hexdigest()


def prefix_before_assistant_turn(
    messages: list[dict[str, Any]], assistant_turn_index: int
) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    seen = 0
    for message in deepcopy(messages):
        if str(message.get("role")) in {"assistant", "MessageRole.ASSISTANT"}:
            if seen == assistant_turn_index:
                break
            seen += 1
        prefix.append(message)
    if seen != assistant_turn_index:
        raise ValueError(f"Assistant turn {assistant_turn_index} is unavailable in trajectory messages")
    return prefix


def to_chat_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    normalized = []
    aliases = {
        "MessageRole.SYSTEM": "system",
        "MessageRole.USER": "user",
        "MessageRole.ASSISTANT": "assistant",
        "MessageRole.TOOL_CALL": "tool-call",
        "MessageRole.TOOL_RESPONSE": "tool-response",
    }
    for message in messages:
        item = deepcopy(message)
        item["role"] = aliases.get(str(item.get("role")), str(item.get("role")))
        normalized.append(ChatMessage.from_dict(item))
    return normalized


def build_teacher_repair_messages(
    *,
    prefix: list[dict[str, Any]],
) -> list[ChatMessage]:
    """Use exactly the trajectory prefix; diagnostics stay outside teacher context."""
    return to_chat_messages(prefix)


def build_repair_sft_messages(
    prefix: list[dict[str, Any]], repaired_output: str
) -> list[dict[str, str]]:
    raw = deepcopy(prefix)
    raw.append({"role": "assistant", "content": repaired_output})
    return prepare_sft_messages(raw)


__all__ = [
    "REPAIR_PROMPT_VERSION",
    "build_repair_sft_messages",
    "build_teacher_repair_messages",
    "prefix_before_assistant_turn",
    "repair_prompt_sha256",
    "to_chat_messages",
]
