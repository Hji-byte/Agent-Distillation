"""Build S0 self-success and full recovery-suffix SFT datasets.

Repair prefixes remain in the conversation as context but receive no loss.
The teacher correction and every verified student continuation action through
the terminal final_answer are supervised. Natural S0 successes supervise every
assistant action and explicitly exclude max-step fallback successes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.train_utils.message_utils import prepare_sft_messages


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _assistant_count(messages: list[dict[str, Any]]) -> int:
    return sum(message.get("role") == "assistant" for message in messages)


def _task_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"]).strip()
    raise ValueError("Trajectory has no non-empty user task")


def _observation_message(observation: str, call_number: int) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"Call id: call_{call_number}\nObservation:\n{observation}",
    }


def build_recovery_example(outcome: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one verified correction plus its complete successful suffix."""
    if not outcome.get("accepted"):
        raise ValueError("Cannot build a recovery example from a rejected outcome")
    attempts = outcome.get("attempts") or []
    selected_index = int(outcome["selected_attempt_index"])
    if not 0 <= selected_index < len(attempts):
        raise ValueError("selected_attempt_index is outside attempts")
    attempt = attempts[selected_index]
    verification = attempt.get("verification") or {}
    trace = verification.get("trace") or []
    if not verification.get("correct") or not trace:
        raise ValueError("Accepted repair lacks a verified successful trace")
    if not trace[-1].get("execution", {}).get("is_final_answer"):
        raise ValueError("Verified repair trace does not end in an explicit final_answer")

    messages = deepcopy(attempt.get("sft_messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Repair SFT prefix must end in the teacher correction")
    supervised_start = _assistant_count(messages) - 1
    if supervised_start < 0:
        raise ValueError("Repair SFT prefix has no assistant correction")

    first_output = str((trace[0].get("action") or {}).get("model_output") or "")
    if messages[-1].get("content") != first_output:
        raise ValueError("Teacher correction differs between sft_messages and verification trace")

    for trace_index, item in enumerate(trace):
        action = item.get("action") or {}
        execution = item.get("execution") or {}
        if trace_index > 0:
            model_output = str(action.get("model_output") or "")
            if not model_output:
                raise ValueError("Continuation trace contains an empty assistant action")
            messages.append({"role": "assistant", "content": model_output})
        is_final = bool(execution.get("is_final_answer"))
        if is_final:
            if trace_index != len(trace) - 1:
                raise ValueError("final_answer appears before the end of the verification trace")
            continue
        observation = execution.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("Non-terminal repair action lacks an execution observation")
        call_number = supervised_start + trace_index + 1
        messages.append(_observation_message(observation, call_number))

    if messages[-1].get("role") != "assistant":
        raise ValueError("Recovery trajectory does not end in an assistant action")
    if _assistant_count(messages) != supervised_start + len(trace):
        raise ValueError("Recovery assistant count does not match verification trace")

    return {
        "schema_version": "local-recovery-sft-v1",
        "messages": messages,
        "supervision": "assistant_suffix",
        "supervised_assistant_start_index": supervised_start,
        "metadata": {
            "source": "repaired_success",
            "repair_id": str(outcome["repair_id"]),
            "run_tag": (outcome.get("experiment_config") or {}).get("run_tag"),
            "failure_kind": outcome.get("failure_kind"),
            "repair_step_index": outcome.get("selected_step_index"),
            "verification_mode": verification.get("verification_mode"),
            "continuation_step_count": verification.get("continuation_step_count"),
            "verified_correct": True,
            "explicit_final_answer": True,
        },
    }


def build_self_success_example(row: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    """Convert one naturally terminated, correctly scored S0 trajectory."""
    metadata = (row.get("log_data") or {}).get("metadata") or {}
    if row.get("score") not in (1, True):
        raise ValueError("Self-success row is not scored correct")
    if str(metadata.get("state")) != "success":
        raise ValueError("Self-success row did not terminate naturally")
    if metadata.get("success") is False:
        raise ValueError("Self-success metadata marks the trajectory unsuccessful")
    validation = metadata.get("trajectory_validation") or {}
    if validation.get("valid") is False:
        raise ValueError("Self-success trajectory failed structural validation")
    raw_messages = (row.get("log_data") or {}).get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Self-success row has no logged messages")
    messages = prepare_sft_messages(raw_messages)
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Self-success trajectory does not end in an assistant action")
    if "final_answer(" not in str(messages[-1].get("content", "")):
        raise ValueError("Self-success final assistant action does not call final_answer")

    task_id = metadata.get("task_id")
    return {
        "schema_version": "local-self-success-sft-v1",
        "messages": messages,
        "supervision": "all_assistant_turns",
        "metadata": {
            "source": "self_success",
            "source_label": source_label,
            "task_id": task_id,
            "state": "success",
            "verified_correct": True,
            "explicit_final_answer": True,
        },
    }


def build_datasets(
    *,
    eligible_repairs_path: Path,
    repair_attempts: list[tuple[str, Path]],
    success_scored: list[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible_keys = {
        (
            str((row.get("metadata") or {}).get("run_tag")),
            str((row.get("metadata") or {}).get("repair_id")),
        )
        for row in _read_jsonl(eligible_repairs_path)
    }
    if (None, None) in eligible_keys or ("None", "None") in eligible_keys:
        raise ValueError("Eligible repair file contains rows without run_tag/repair_id")

    latest_outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    repair_inputs = []
    for expected_run_tag, path in repair_attempts:
        repair_inputs.append({"run_tag": expected_run_tag, "path": str(path.resolve()), "sha256": _sha256(path)})
        for outcome in _read_jsonl(path):
            actual_run_tag = str((outcome.get("experiment_config") or {}).get("run_tag"))
            if actual_run_tag != expected_run_tag:
                raise ValueError(
                    f"Run tag mismatch in {path}: expected {expected_run_tag!r}, got {actual_run_tag!r}"
                )
            latest_outcomes[(actual_run_tag, str(outcome.get("repair_id")))] = outcome

    missing = sorted(eligible_keys - set(latest_outcomes))
    if missing:
        raise ValueError(f"Missing {len(missing)} eligible repairs, first keys: {missing[:5]}")
    repair_rows = [
        build_recovery_example(latest_outcomes[key]) for key in sorted(eligible_keys)
    ]

    self_rows: list[dict[str, Any]] = []
    self_rejections: Counter[str] = Counter()
    success_inputs = []
    for label, path in success_scored:
        success_inputs.append({"label": label, "path": str(path.resolve()), "sha256": _sha256(path)})
        for row in _read_jsonl(path):
            metadata = (row.get("log_data") or {}).get("metadata") or {}
            if row.get("score") not in (1, True):
                self_rejections["incorrect"] += 1
                continue
            if str(metadata.get("state")) != "success":
                self_rejections[f"state:{metadata.get('state')}"] += 1
                continue
            validation = metadata.get("trajectory_validation") or {}
            if validation.get("valid") is False:
                self_rejections["invalid_trajectory"] += 1
                continue
            self_rows.append(build_self_success_example(row, source_label=label))

    all_rows = repair_rows + self_rows
    seen_tasks: dict[str, str] = {}
    for row in all_rows:
        task = _task_text(row["messages"])
        source = str((row.get("metadata") or {}).get("source"))
        if task in seen_tasks:
            raise ValueError(
                f"Duplicate task across {seen_tasks[task]} and {source}: {task[:120]!r}"
            )
        seen_tasks[task] = source

    summary = {
        "schema_version": "recovery-training-build-v1",
        "status": "complete" if success_scored else "incomplete_missing_self_success_inputs",
        "eligible_repairs_path": str(eligible_repairs_path.resolve()),
        "eligible_repairs_sha256": _sha256(eligible_repairs_path),
        "repair_inputs": repair_inputs,
        "success_inputs": success_inputs,
        "counts": {
            "repaired_success": len(repair_rows),
            "self_success": len(self_rows),
            "combined": len(all_rows),
            "self_success_rejections": dict(sorted(self_rejections.items())),
        },
        "repair_failure_kinds": dict(
            sorted(Counter(row["metadata"]["failure_kind"] for row in repair_rows).items())
        ),
        "repair_verification_modes": dict(
            sorted(Counter(row["metadata"]["verification_mode"] for row in repair_rows).items())
        ),
    }
    return repair_rows, self_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-repairs", type=Path, required=True)
    parser.add_argument(
        "--repair-attempt",
        action="append",
        default=[],
        type=_parse_labeled_path,
        metavar="RUN_TAG=PATH",
    )
    parser.add_argument(
        "--success-scored",
        action="append",
        default=[],
        type=_parse_labeled_path,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-repairs", type=int)
    parser.add_argument("--expected-self-success", type=int)
    args = parser.parse_args()
    if not args.repair_attempt:
        parser.error("At least one --repair-attempt is required")

    repair_rows, self_rows, summary = build_datasets(
        eligible_repairs_path=args.eligible_repairs,
        repair_attempts=args.repair_attempt,
        success_scored=args.success_scored,
    )
    if args.expected_repairs is not None and len(repair_rows) != args.expected_repairs:
        raise ValueError(
            f"Expected {args.expected_repairs} repairs, constructed {len(repair_rows)}"
        )
    if args.expected_self_success is not None and len(self_rows) != args.expected_self_success:
        raise ValueError(
            f"Expected {args.expected_self_success} self-successes, constructed {len(self_rows)}"
        )

    repair_path = args.output_dir / "repaired_success_full_suffix.jsonl"
    self_path = args.output_dir / "self_success_natural.jsonl"
    combined_name = (
        "onpolicy_success_combined.jsonl"
        if args.success_scored
        else "onpolicy_success_combined_INCOMPLETE.jsonl"
    )
    combined_path = args.output_dir / combined_name
    _write_jsonl(repair_path, repair_rows)
    _write_jsonl(self_path, self_rows)
    _write_jsonl(combined_path, [*repair_rows, *self_rows])
    summary["outputs"] = {
        "repair": {"path": str(repair_path.resolve()), "sha256": _sha256(repair_path)},
        "self_success": {"path": str(self_path.resolve()), "sha256": _sha256(self_path)},
        "combined": {"path": str(combined_path.resolve()), "sha256": _sha256(combined_path)},
    }
    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
