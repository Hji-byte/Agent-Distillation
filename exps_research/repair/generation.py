"""Strict one-action generation shared by teacher repair and verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model

from .execution import parse_action


@dataclass
class GeneratedAction:
    model_output: str
    code_action: str
    format_retry_count: int
    input_tokens: int
    output_tokens: int

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class StrictActionGenerator:
    """Generate exactly one Thought/Code action, with an optional format retry."""

    def __init__(self, model: Model, max_format_retries: int = 1):
        if max_format_retries < 0:
            raise ValueError("max_format_retries must be non-negative")
        self.model = model
        self.max_format_retries = max_format_retries

    def generate(self, messages: list[ChatMessage]) -> GeneratedAction:
        request_messages = list(messages)
        total_input = 0
        total_output = 0
        last_error: Exception | None = None

        for attempt in range(self.max_format_retries + 1):
            response = self.model.generate(
                request_messages,
                stop_sequences=["Observation:", "Calling tools:", "</code>"],
            )
            if response.token_usage:
                total_input += response.token_usage.input_tokens
                total_output += response.token_usage.output_tokens
            text = str(response.content or "")
            # Native CodeAgent also restores the closing stop tag after model
            # generation. Only do this when an opening tag actually exists.
            if "<code>" in text and not text.rstrip().endswith("</code>"):
                text = text.rstrip() + "\n</code>"
            try:
                code = parse_action(text)
                return GeneratedAction(
                    model_output=text,
                    code_action=code,
                    format_retry_count=attempt,
                    input_tokens=total_input,
                    output_tokens=total_output,
                )
            except Exception as error:
                last_error = error
                if attempt >= self.max_format_retries:
                    break
                request_messages = list(messages) + [
                    ChatMessage(role=MessageRole.ASSISTANT, content=text),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "Regenerate only the current action. Output a non-empty `Thought:` line "
                            "followed by exactly one complete `<code>...</code>` block. Do not explain "
                            "the format correction."
                        ),
                    ),
                ]

        raise ValueError(f"Model failed to produce a valid Thought/Code action: {last_error}")


__all__ = ["GeneratedAction", "StrictActionGenerator"]
