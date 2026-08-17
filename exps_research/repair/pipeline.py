"""Verifier-grounded, error-aware backward repair pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from smolagents.models import ChatMessage, MessageRole, Model

from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal
from exps_research.unified_framework.math_utils.qwen_math_parser import extract_answer

from .execution import DEFAULT_REPAIR_IMPORTS, IsolatedReplay
from .generation import StrictActionGenerator
from .localization import classify_failure, error_aware_backward_candidates
from .messages import (
    build_repair_sft_messages,
    build_teacher_repair_messages,
    prefix_before_assistant_turn,
    to_chat_messages,
)


@dataclass
class RepairConfig:
    max_candidates: int = 5
    max_continuation_steps: int = 4
    max_format_retries: int = 1
    execution_timeout_seconds: int = 30
    additional_authorized_imports: list[str] | None = None

    def imports(self) -> list[str]:
        return list(self.additional_authorized_imports or DEFAULT_REPAIR_IMPORTS)


def _stable_id(entry: dict[str, Any]) -> str:
    source_id = ((entry.get("log_data") or {}).get("metadata") or {}).get("task_id")
    if source_id is not None:
        return str(source_id)
    return hashlib.sha256(str(entry.get("question", "")).encode("utf-8")).hexdigest()[:16]


def _grade_math(answer: Any, gold: Any) -> bool:
    predicted = str(answer)
    parsed = extract_answer(predicted) if "boxed" in predicted else predicted
    return bool(math_equal(parsed, str(gold), timeout=True))


def _append_execution_messages(
    messages: list[ChatMessage],
    *,
    model_output: str,
    observation: str,
    call_index: int,
) -> None:
    call_id = f"repair_call_{call_index}"
    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=model_output))
    messages.append(
        ChatMessage(
            role=MessageRole.TOOL_CALL,
            content=f"Calling tools:\n[{{'id': '{call_id}', 'function': {{'name': 'python_interpreter'}}}}]",
        )
    )
    messages.append(
        ChatMessage(
            role=MessageRole.TOOL_RESPONSE,
            content=f"Call id: {call_id}\nObservation:\n{observation}",
        )
    )


class RepairPipeline:
    def __init__(
        self,
        *,
        teacher_model: Model,
        continuation_model: Model,
        config: RepairConfig | None = None,
    ):
        self.config = config or RepairConfig()
        self.teacher = StrictActionGenerator(teacher_model, self.config.max_format_retries)
        self.continuation = StrictActionGenerator(continuation_model, self.config.max_format_retries)
        self.teacher_model_id = getattr(teacher_model, "model_id", type(teacher_model).__name__)
        self.continuation_model_id = getattr(continuation_model, "model_id", type(continuation_model).__name__)

    def _verify(
        self,
        *,
        prefix: list[dict[str, Any]],
        repaired_action,
        repaired_execution,
        replay: IsolatedReplay,
        gold_answer: Any,
    ) -> dict[str, Any]:
        trace: list[dict[str, Any]] = [
            {
                "source": "teacher_repair",
                "action": repaired_action.dict(),
                "execution": repaired_execution.dict(),
            }
        ]
        if repaired_execution.is_final_answer:
            correct = _grade_math(repaired_execution.output, gold_answer)
            return {
                "correct": correct,
                "final_answer": str(repaired_execution.output),
                "trace": trace,
            }

        messages = to_chat_messages(prefix)
        _append_execution_messages(
            messages,
            model_output=repaired_action.model_output,
            observation=repaired_execution.observation,
            call_index=0,
        )
        for index in range(1, self.config.max_continuation_steps + 1):
            generated = self.continuation.generate(messages)
            try:
                executed = replay.execute_code(generated.code_action)
            except Exception as error:
                return {
                    "correct": False,
                    "final_answer": None,
                    "trace": trace,
                    "rejection_reason": f"continuation_execution_error: {error}",
                }
            trace.append(
                {
                    "source": "continuation_policy",
                    "action": generated.dict(),
                    "execution": executed.dict(),
                }
            )
            if executed.is_final_answer:
                return {
                    "correct": _grade_math(executed.output, gold_answer),
                    "final_answer": str(executed.output),
                    "trace": trace,
                }
            _append_execution_messages(
                messages,
                model_output=generated.model_output,
                observation=executed.observation,
                call_index=index,
            )
        return {
            "correct": False,
            "final_answer": None,
            "trace": trace,
            "rejection_reason": "continuation_reached_step_limit",
        }

    def repair_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        if entry.get("score") not in (0, False):
            raise ValueError("Repair input must be a scored failed trajectory (score == 0)")
        log_data = entry.get("log_data") or {}
        messages = log_data.get("messages") or []
        steps = log_data.get("trajectory_steps") or []
        if not messages or not steps:
            raise ValueError("Repair input is missing messages or trajectory_steps")
        gold = entry.get("true_answer", entry.get("answer"))
        if gold is None:
            raise ValueError("Repair input has no gold answer")

        candidates = error_aware_backward_candidates(entry)[: self.config.max_candidates]
        outcome: dict[str, Any] = {
            "schema_version": "local-repair-v1",
            "repair_id": _stable_id(entry),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "question": entry.get("question"),
            "gold_answer": str(gold),
            "original_generated_answer": entry.get("generated_answer"),
            "original_score": entry.get("score"),
            "failure_kind": classify_failure(entry),
            "candidate_order": [candidate.dict() for candidate in candidates],
            "teacher_model_id": self.teacher_model_id,
            "continuation_model_id": self.continuation_model_id,
            "config": asdict(self.config),
            "attempts": [],
            "accepted": False,
        }

        for candidate in candidates:
            step = steps[candidate.step_index]
            attempt: dict[str, Any] = {"candidate": candidate.dict(), "accepted": False}
            try:
                prefix = prefix_before_assistant_turn(messages, candidate.assistant_turn_index)
                replay = IsolatedReplay(
                    self.config.imports(),
                    timeout_seconds=self.config.execution_timeout_seconds,
                )
                replayed = replay.replay_prefix(steps, candidate.step_index)
                attempt["replayed_prefix_steps"] = [item.dict() for item in replayed]

                teacher_messages = build_teacher_repair_messages(
                    prefix=prefix,
                    failure_kind=candidate.failure_kind,
                    failed_output=step.get("model_output"),
                    failed_observation=step.get("error") or step.get("observations"),
                )
                repaired = self.teacher.generate(teacher_messages)
                attempt["teacher_action"] = repaired.dict()
                repaired_execution = replay.execute_code(repaired.code_action)
                attempt["teacher_execution"] = repaired_execution.dict()
                verification = self._verify(
                    prefix=prefix,
                    repaired_action=repaired,
                    repaired_execution=repaired_execution,
                    replay=replay,
                    gold_answer=gold,
                )
                attempt["verification"] = verification
                if verification["correct"]:
                    attempt["accepted"] = True
                    attempt["sft_messages"] = build_repair_sft_messages(prefix, repaired.model_output)
                    attempt["target_assistant_turn_index"] = sum(
                        message["role"] == "assistant" for message in attempt["sft_messages"]
                    ) - 1
                    outcome["accepted"] = True
                    outcome["selected_step_index"] = candidate.step_index
                    outcome["selected_attempt_index"] = len(outcome["attempts"])
                    outcome["attempts"].append(attempt)
                    break
            except Exception as error:
                attempt["rejection_reason"] = f"{type(error).__name__}: {error}"
            outcome["attempts"].append(attempt)

        if not outcome["accepted"]:
            outcome["rejection_reason"] = "no_candidate_produced_a_verified_correct_trajectory"
        return outcome


__all__ = ["RepairConfig", "RepairPipeline"]
