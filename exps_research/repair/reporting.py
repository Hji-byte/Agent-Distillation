"""Run-level statistics for local repair experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def _repair_id(entry: dict[str, Any]) -> str:
    task_id = (((entry.get("log_data") or {}).get("metadata") or {}).get("task_id"))
    if task_id is not None:
        return str(task_id)
    return hashlib.sha256(str(entry.get("question", "")).encode("utf-8")).hexdigest()[:16]


def summarize_repair_run(
    scored_path: str | Path,
    attempts_path: str | Path,
    *,
    max_entries: int = -1,
) -> dict[str, Any]:
    scored = _rows(scored_path)
    normal_correct = [
        row
        for row in scored
        if row.get("score") in (1, True)
        and str(((row.get("log_data") or {}).get("metadata") or {}).get("state")) == "success"
    ]
    fallback_correct = [
        row
        for row in scored
        if row.get("score") in (1, True)
        and str(((row.get("log_data") or {}).get("metadata") or {}).get("state")) == "max_steps_error"
    ]
    raw_attempts = _rows(attempts_path)
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_attempts:
        latest_by_id[str(row["repair_id"])] = row
    attempts = list(latest_by_id.values())
    expected_failure_ids = {
        _repair_id(row)
        for row in scored
        if row.get("score") in (0, False) and row.get("log_data")
    }
    accepted = [row for row in attempts if row.get("accepted")]
    retryable = [row for row in attempts if row.get("retryable_error")]
    scientific_rejected = [
        row for row in attempts if not row.get("accepted") and not row.get("retryable_error")
    ]
    completed_ids = {
        str(row["repair_id"])
        for row in attempts
        if row.get("accepted") or not row.get("retryable_error")
    }
    remaining_ids = sorted(expected_failure_ids - completed_ids)
    if not remaining_ids:
        completion_status = "complete"
    elif max_entries > 0:
        completion_status = "partial_by_entry_limit"
    elif retryable:
        completion_status = "incomplete_retryable_errors"
    else:
        completion_status = "incomplete"
    attempted_candidates = [attempt for row in attempts for attempt in row.get("attempts", [])]
    teacher_actions = [
        attempt["teacher_action"]
        for attempt in attempted_candidates
        if attempt.get("teacher_action")
    ]
    continuation_actions = [
        trace["action"]
        for attempt in attempted_candidates
        for trace in (attempt.get("verification") or {}).get("trace", [])
        if trace.get("source") == "continuation_policy" and trace.get("action")
    ]
    mode_counts = {"teacher_terminal": 0, "student_continuation": 0}
    for row in accepted:
        mode = row.get("verification_mode")
        if mode in mode_counts:
            mode_counts[mode] += 1

    def total(actions: list[dict[str, Any]], field: str) -> int:
        return sum(int(action.get(field) or 0) for action in actions)

    return {
        "schema_version": "local-repair-summary-v1",
        "s0": {
            "total": len(scored),
            "correct": sum(row.get("score") in (1, True) for row in scored),
            "failed": sum(row.get("score") in (0, False) for row in scored),
            "normal_success_correct": len(normal_correct),
            "max_steps_fallback_correct": len(fallback_correct),
        },
        "repair": {
            "completion_status": completion_status,
            "expected_failures": len(expected_failure_ids),
            "processed": len(attempts),
            "raw_attempt_records": len(raw_attempts),
            "completed": len(completed_ids & expected_failure_ids),
            "remaining": len(remaining_ids),
            "remaining_repair_ids": remaining_ids,
            "accepted": len(accepted),
            "scientific_rejected": len(scientific_rejected),
            "retryable_error": len(retryable),
            "average_candidates_attempted": (
                sum(len(row.get("attempts", [])) for row in attempts) / len(attempts)
                if attempts
                else 0.0
            ),
            "completion_mode": mode_counts,
            "average_continuation_steps_accepted": (
                sum(int(row.get("continuation_step_count") or 0) for row in accepted) / len(accepted)
                if accepted
                else 0.0
            ),
        },
        "retries": {
            "teacher_format_retries": total(teacher_actions, "format_retry_count"),
            "continuation_format_retries": total(continuation_actions, "format_retry_count"),
        },
        "token_usage": {
            "s0_evaluation_input_tokens": sum(int(row.get("input_tokens") or 0) for row in scored),
            "s0_evaluation_output_tokens": sum(int(row.get("output_tokens") or 0) for row in scored),
            "teacher_input_tokens": total(teacher_actions, "input_tokens"),
            "teacher_output_tokens": total(teacher_actions, "output_tokens"),
            "continuation_input_tokens": total(continuation_actions, "input_tokens"),
            "continuation_output_tokens": total(continuation_actions, "output_tokens"),
        },
    }


__all__ = ["summarize_repair_run"]
