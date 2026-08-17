#!/usr/bin/env python
# coding=utf-8

"""Structured trajectory validation for Agent-distillation datasets.

This module is a project extension to smolagents 1.26. It deliberately works
with both live ``RunResult``/``ActionStep`` objects and their serialized dicts
so validation uses framework state instead of matching rendered error text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


_THOUGHT_RE = re.compile(r"(?m)^\s*Thought:\s*\S")
_CODE_OPEN_RE = re.compile(r"(?m)^\s*<code>[ \t]*\r?\n")
_CODE_CLOSE_RE = re.compile(r"</code>\s*$")
_FINAL_ANSWER_RE = re.compile(r"\bfinal_answer\s*\(")


@dataclass
class TrajectoryValidationReport:
    """Result of validating one structured smolagents run for SFT."""

    valid: bool = False
    reasons: list[str] = field(default_factory=list)
    parsing_error_assistant_turns: list[int] = field(default_factory=list)
    execution_error_assistant_turns: list[int] = field(default_factory=list)
    other_error_assistant_turns: list[int] = field(default_factory=list)

    def dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "parsing_error_assistant_turns": list(self.parsing_error_assistant_turns),
            "execution_error_assistant_turns": list(self.execution_error_assistant_turns),
            "other_error_assistant_turns": list(self.other_error_assistant_turns),
        }


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _serialized_step(step: Any) -> Mapping[str, Any]:
    if isinstance(step, Mapping):
        return step
    dict_method = getattr(step, "dict", None)
    if callable(dict_method):
        serialized = dict_method()
        if isinstance(serialized, Mapping):
            return serialized
    raise TypeError(f"Unsupported trajectory step type: {type(step).__name__}")


def _has_complete_action_format(model_output: Any) -> bool:
    text = "" if model_output is None else str(model_output).strip()
    return bool(_THOUGHT_RE.search(text) and _CODE_OPEN_RE.search(text) and _CODE_CLOSE_RE.search(text))


def _error_kind(error: Any) -> str | None:
    if error is None:
        return None
    error_type = str(_read(error, "type", type(error).__name__))
    message = str(_read(error, "message", error))
    normalized = re.sub(r"[^a-z]", "", f"{error_type} {message}".lower())
    if "parsing" in normalized:
        return "parsing"
    if "maxsteps" in normalized:
        return "max_steps"
    if "execution" in normalized or "toolcall" in normalized:
        return "execution"
    return "other"


def validate_run_result_for_sft(run_result: Any) -> TrajectoryValidationReport:
    """Validate a structured run according to the project SFT policy.

    Policy:
    - the run state must be ``success``;
    - every Assistant action must contain a complete Thought/Code block;
    - parsing and unknown framework errors reject the trajectory;
    - execution errors remain valid recovery context and are linked to the
      Assistant turn that produced them;
    - the last Assistant action must be marked final and call
      ``final_answer(...)`` explicitly.
    """
    report = TrajectoryValidationReport()
    state = str(_read(run_result, "state", "unknown"))
    raw_steps = _read(run_result, "steps")
    if raw_steps is None:
        report.reasons.append("missing_structured_run_result")
        return report
    if state != "success":
        report.reasons.append("state_not_success")

    action_steps: list[tuple[int, Mapping[str, Any]]] = []
    assistant_turn = 0
    for raw_step in raw_steps:
        step = _serialized_step(raw_step)
        if step.get("model_output") is None:
            continue
        explicit_turn = step.get("assistant_turn_index")
        turn = int(explicit_turn) if explicit_turn is not None else assistant_turn
        action_steps.append((turn, step))
        assistant_turn += 1

    if not action_steps:
        report.reasons.append("no_assistant_actions")

    for turn, step in action_steps:
        if not _has_complete_action_format(step.get("model_output")):
            report.reasons.append("incomplete_action_format")
        error_kind = _error_kind(step.get("error"))
        if error_kind == "parsing":
            report.parsing_error_assistant_turns.append(turn)
        elif error_kind == "execution":
            report.execution_error_assistant_turns.append(turn)
        elif error_kind in {"max_steps", "other"}:
            report.other_error_assistant_turns.append(turn)

    if report.parsing_error_assistant_turns:
        report.reasons.append("parsing_error")
    if report.other_error_assistant_turns:
        report.reasons.append("non_recoverable_step_error")

    if action_steps:
        _, final_step = action_steps[-1]
        final_output = str(final_step.get("model_output") or "")
        if not final_step.get("is_final_answer") or not _FINAL_ANSWER_RE.search(final_output):
            report.reasons.append("missing_final_answer")

    report.reasons = list(dict.fromkeys(report.reasons))
    report.valid = not report.reasons
    return report


__all__ = [
    "TrajectoryValidationReport",
    "validate_run_result_for_sft",
]
