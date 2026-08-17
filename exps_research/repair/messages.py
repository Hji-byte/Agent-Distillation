"""Message construction for one-step teacher repair and SFT output."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from smolagents.models import ChatMessage

from exps_research.train_utils.message_utils import prepare_sft_messages


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(content_text(block) for block in content)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content)


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


def repair_instruction(
    *,
    failure_kind: str,
    failed_output: Any,
    failed_observation: Any,
) -> str:
    return f"""The student failed at the current agent step.

Failure type: {failure_kind}

Student's rejected current action:
<rejected_action>
{content_text(failed_output)}
</rejected_action>

Observation or framework error associated with that action:
<rejected_observation>
{content_text(failed_observation)}
</rejected_observation>

Replace only this current assistant action. Preserve the task and interaction
state before it. Do not restart the whole solution, do not provide later
steps, and do not mention that you are correcting the student.

Output exactly one complete action in this form:
Thought: <concise reasoning for this action>
<code>
# executable Python action
</code>"""


def build_teacher_repair_messages(
    *,
    prefix: list[dict[str, Any]],
    failure_kind: str,
    failed_output: Any,
    failed_observation: Any,
) -> list[ChatMessage]:
    messages = to_chat_messages(prefix)
    messages.append(
        ChatMessage.from_dict(
            {
                "role": "user",
                "content": repair_instruction(
                    failure_kind=failure_kind,
                    failed_output=failed_output,
                    failed_observation=failed_observation,
                ),
            }
        )
    )
    return messages


def build_repair_sft_messages(
    prefix: list[dict[str, Any]], repaired_output: str
) -> list[dict[str, str]]:
    raw = deepcopy(prefix)
    raw.append({"role": "assistant", "content": repaired_output})
    return prepare_sft_messages(raw)


__all__ = [
    "build_repair_sft_messages",
    "build_teacher_repair_messages",
    "content_text",
    "prefix_before_assistant_turn",
    "repair_instruction",
    "to_chat_messages",
]
