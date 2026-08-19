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
    generation_attempts: list[dict[str, Any]]

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionGenerationError(ValueError):
    """A format-generation failure that retains every billed model response."""

    def __init__(
        self,
        message: str,
        *,
        generation_attempts: list[dict[str, Any]],
        generation_source: str,
    ):
        super().__init__(message)
        self.generation_attempts = generation_attempts
        self.generation_source = generation_source

    @property
    def format_retry_count(self) -> int:
        return max(0, len(self.generation_attempts) - 1)

    @property
    def input_tokens(self) -> int:
        return sum(int(item.get("input_tokens") or 0) for item in self.generation_attempts)

    @property
    def output_tokens(self) -> int:
        return sum(int(item.get("output_tokens") or 0) for item in self.generation_attempts)

    def dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "generation_source": self.generation_source,
            "format_retry_count": self.format_retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_attempts": self.generation_attempts,
        }


class StrictActionGenerator:
    """Generate exactly one Thought/Code action, with an optional format retry."""

    def __init__(
        self,
        model: Model,
        max_format_retries: int = 1,
        generation_source: str = "action",
    ):
        if max_format_retries < 0:
            raise ValueError("max_format_retries must be non-negative")
        self.model = model
        self.max_format_retries = max_format_retries
        self.generation_source = generation_source

    def generate(self, messages: list[ChatMessage]) -> GeneratedAction:
        request_messages = list(messages)
        total_input = 0
        total_output = 0
        last_error: Exception | None = None
        generation_attempts: list[dict[str, Any]] = []

        for attempt in range(self.max_format_retries + 1):
            response = self.model.generate(
                request_messages,
                stop_sequences=["Observation:", "Calling tools:", "</code>"],
            )
            if response.token_usage:
                total_input += response.token_usage.input_tokens
                total_output += response.token_usage.output_tokens
            raw_text = str(response.content or "")
            text = raw_text
            # Native CodeAgent also restores the closing stop tag after model
            # generation. Only do this when an opening tag actually exists.
            closing_tag_restored = False
            if "<code>" in text and not text.rstrip().endswith("</code>"):
                text = text.rstrip() + "\n</code>"
                closing_tag_restored = True
            generation_attempt = {
                "attempt_index": attempt,
                "raw_model_output": raw_text,
                "model_output": text,
                "closing_tag_restored": closing_tag_restored,
                "finish_reason": self._finish_reason(response),
                "parse_error": None,
                "valid_action": False,
                "input_tokens": response.token_usage.input_tokens if response.token_usage else 0,
                "output_tokens": response.token_usage.output_tokens if response.token_usage else 0,
            }
            try:
                code = parse_action(text)
                generation_attempt["valid_action"] = True
                generation_attempts.append(generation_attempt)
                return GeneratedAction(
                    model_output=text,
                    code_action=code,
                    format_retry_count=attempt,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    generation_attempts=generation_attempts,
                )
            except Exception as error:
                last_error = error
                generation_attempt["parse_error"] = f"{type(error).__name__}: {error}"
                generation_attempts.append(generation_attempt)
                if attempt >= self.max_format_retries:
                    break
                request_messages = list(messages) + [
                    ChatMessage(role=MessageRole.ASSISTANT, content=text),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Your previous action omitted the mandatory non-empty line beginning exactly "
                                    "`Thought:`. Regenerate only the current assistant action. "
                                    "Start with `Thought: <reasoning>`, then provide the complete "
                                    "<code>...</code> action. Do not discuss the correction."
                                ),
                            }
                        ],
                    ),
                ]

        raise ActionGenerationError(
            f"Model failed to produce a valid Thought/Code action: {last_error}",
            generation_attempts=generation_attempts,
            generation_source=self.generation_source,
        )

    @staticmethod
    def _finish_reason(response: ChatMessage) -> str | None:
        """Extract OpenAI-compatible termination metadata without storing raw API objects."""
        raw = getattr(response, "raw", None)
        choices = getattr(raw, "choices", None)
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)


__all__ = ["ActionGenerationError", "GeneratedAction", "StrictActionGenerator"]
