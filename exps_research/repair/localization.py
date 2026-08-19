"""Error-aware backward candidate selection for local trajectory repair."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Any


_THOUGHT_RE = re.compile(r"(?m)^[ \t]*Thought:[ \t]*\S")
_CODE_RE = re.compile(r"(?s)<code>.*?</code>\s*$")


@dataclass(frozen=True)
class RepairCandidate:
    step_index: int
    assistant_turn_index: int
    failure_kind: str
    reason: str

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def _error_kind(error: Any) -> str | None:
    if not error:
        return None
    if isinstance(error, dict):
        text = f"{error.get('type', '')} {error.get('message', '')}"
    else:
        text = str(error)
    normalized = re.sub(r"[^a-z]", "", text.lower())
    if "parsing" in normalized:
        return "parsing_error"
    if "execution" in normalized or "interpreter" in normalized or "toolcall" in normalized:
        return "execution_error"
    if "maxsteps" in normalized:
        return "max_steps"
    return "other_error"


def _complete_action(output: Any) -> bool:
    text = str(output or "").strip()
    return bool(_THOUGHT_RE.search(text) and _CODE_RE.search(text))


def _forced_max_steps_fallback(step: dict[str, Any]) -> bool:
    """Identify smolagents' post-budget plain-text fallback, not an agent action."""
    return (
        step.get("model_output") is None
        and step.get("code_action") is None
        and _error_kind(step.get("error")) == "max_steps"
    )


def _pure_final_answer(code: Any) -> bool:
    """Return whether code only submits an already-computed value."""
    if not code:
        return False
    try:
        tree = ast.parse(str(code))
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return False
    call = tree.body[0].value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "final_answer"
    )


def classify_failure(entry: dict[str, Any]) -> str:
    log_data = entry.get("log_data") or {}
    steps = log_data.get("trajectory_steps") or []
    for step in steps:
        if _forced_max_steps_fallback(step):
            continue
        if not _complete_action(step.get("model_output")):
            return "incomplete_action_format"
        if kind := _error_kind(step.get("error")):
            return kind
    state = str((log_data.get("metadata") or {}).get("state", "unknown"))
    if state != "success":
        return "max_steps" if state == "max_steps_error" else "state_not_success"
    if not steps or not steps[-1].get("is_final_answer"):
        return "missing_final_answer"
    if entry.get("score") in (0, False):
        return "wrong_answer"
    return "unknown_failure"


def error_aware_backward_candidates(entry: dict[str, Any]) -> list[RepairCandidate]:
    """Return candidate steps in error-aware, minimal-backward order.

    Explicit format/parsing/execution errors are tried first. Otherwise the
    search starts at the last substantive computation step and moves backward.
    A pure ``final_answer(variable)`` step is deferred for wrong-answer cases,
    because rewriting it often hides an upstream computation error.
    """
    log_data = entry.get("log_data") or {}
    steps = log_data.get("trajectory_steps") or []
    if not steps:
        return []

    explicit: list[tuple[int, str]] = []
    repairable_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if not _forced_max_steps_fallback(step)
    ]
    for index, step in repairable_steps:
        if not _complete_action(step.get("model_output")):
            explicit.append((index, "incomplete_action_format"))
        elif kind := _error_kind(step.get("error")):
            explicit.append((index, kind))

    overall_kind = classify_failure(entry)
    ordered: list[tuple[int, str, str]] = []
    if explicit:
        # The closest explicit failure to the terminal outcome is the minimal
        # intervention; earlier steps remain fallbacks.
        for index, kind in reversed(explicit):
            ordered.append((index, kind, "explicit_framework_error"))
        earliest_error = min(index for index, _ in explicit)
        for index, _ in reversed(repairable_steps):
            if index < earliest_error:
                ordered.append((index, overall_kind, "backward_fallback"))
    else:
        indices = [index for index, _ in reversed(repairable_steps)]
        last_repairable_index = repairable_steps[-1][0] if repairable_steps else None
        if (
            overall_kind == "wrong_answer"
            and last_repairable_index is not None
            and _pure_final_answer(steps[last_repairable_index].get("code_action"))
        ):
            indices = [index for index in indices if index != last_repairable_index] + [
                last_repairable_index
            ]
        for position, index in enumerate(indices):
            reason = (
                "answer_submission_fallback"
                if position == len(indices) - 1 and index == last_repairable_index
                else "backward_search"
            )
            ordered.append((index, overall_kind, reason))

    candidates: list[RepairCandidate] = []
    seen: set[int] = set()
    for index, kind, reason in ordered:
        if index in seen:
            continue
        seen.add(index)
        turn = steps[index].get("assistant_turn_index")
        candidates.append(
            RepairCandidate(
                step_index=index,
                assistant_turn_index=int(turn if turn is not None else index),
                failure_kind=kind,
                reason=reason,
            )
        )
    return candidates


__all__ = ["RepairCandidate", "classify_failure", "error_aware_backward_candidates"]
