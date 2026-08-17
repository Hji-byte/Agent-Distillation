"""Filter scored trajectories using the smolagents fork's validation API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from smolagents import validate_run_result_for_sft

from .path_utils import bounded_artifact_path


def _load_log_data(result: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    log_data = result.get("log_data")
    if not log_data:
        return None, False
    if isinstance(log_data, str):
        try:
            decoded = json.loads(log_data)
        except json.JSONDecodeError:
            return None, True
        return (decoded, True) if isinstance(decoded, dict) else (None, True)
    return (log_data, False) if isinstance(log_data, dict) else (None, False)


def inspect_agent_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    """Combine deterministic answer scoring with framework validation."""
    log_data, _ = _load_log_data(result)
    if log_data is None:
        return {
            "valid": False,
            "reasons": ["empty_or_invalid_log"],
            "parsing_error_assistant_turns": [],
            "execution_error_assistant_turns": [],
            "other_error_assistant_turns": [],
        }

    metadata = log_data.get("metadata")
    trajectory_steps = log_data.get("trajectory_steps")
    if not isinstance(metadata, dict) or not isinstance(trajectory_steps, list):
        return {
            "valid": False,
            "reasons": ["missing_structured_run_result"],
            "parsing_error_assistant_turns": [],
            "execution_error_assistant_turns": [],
            "other_error_assistant_turns": [],
        }

    report = validate_run_result_for_sft(
        {"state": metadata.get("state", "unknown"), "steps": trajectory_steps}
    ).dict()
    if result.get("score") != 1:
        report["reasons"].append("incorrect_answer")
        report["valid"] = False
    report["reasons"] = list(dict.fromkeys(report["reasons"]))
    return report


def _store_validation(result: dict[str, Any], report: dict[str, Any]) -> None:
    log_data, was_string = _load_log_data(result)
    if log_data is None:
        return
    metadata = log_data.setdefault("metadata", {})
    metadata["trajectory_validation"] = report
    metadata["parsing_error_assistant_turns"] = report["parsing_error_assistant_turns"]
    metadata["execution_error_assistant_turns"] = report["execution_error_assistant_turns"]
    metadata["other_error_assistant_turns"] = report["other_error_assistant_turns"]
    result["log_data"] = json.dumps(log_data, ensure_ascii=False) if was_string else log_data


def filter_agent_trajectories(result_path: str, do_save: bool = True) -> dict[str, int]:
    result_file = Path(result_path)
    if result_file.name.endswith("_scored.jsonl"):
        output_base = result_file.name.removesuffix("_scored.jsonl")
    else:
        output_base = result_file.stem
    output_dir_path = result_file.parent
    if output_dir_path.name == "evaluations":
        output_dir_path = output_dir_path.parent / "filtered_data"
    output_path = bounded_artifact_path(output_dir_path, output_base, "_filtered.jsonl")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print("Save filtered data to", output_path)

    with open(result_path, encoding="utf-8") as result_file:
        results = [json.loads(line) for line in result_file if line.strip()]

    stats = {
        "total": len(results),
        "correct_answers": 0,
        "incorrect_answers": 0,
        "empty_or_invalid_log": 0,
        "missing_structured_run_result": 0,
        "state_not_success": 0,
        "error_parsing": 0,
        "error_execution": 0,
        "error_other": 0,
        "incomplete_action_format": 0,
        "missing_final_answer": 0,
        "valid_entries": 0,
    }
    valid_results = []

    for result in results:
        stats["correct_answers" if result.get("score") == 1 else "incorrect_answers"] += 1
        report = inspect_agent_trajectory(result)
        for reason, stat_name in (
            ("empty_or_invalid_log", "empty_or_invalid_log"),
            ("missing_structured_run_result", "missing_structured_run_result"),
            ("state_not_success", "state_not_success"),
            ("parsing_error", "error_parsing"),
            ("non_recoverable_step_error", "error_other"),
            ("incomplete_action_format", "incomplete_action_format"),
            ("missing_final_answer", "missing_final_answer"),
        ):
            if reason in report["reasons"]:
                stats[stat_name] += 1
        if report["execution_error_assistant_turns"]:
            stats["error_execution"] += 1

        _store_validation(result, report)
        if report["valid"]:
            valid_results.append(result)
            stats["valid_entries"] += 1

    print(f"Original log size: {stats['total']}")
    print(f"Filtered log size: {stats['valid_entries']}")
    print(f"Output path: {output_path}")
    print("\nDetailed statistics:")
    for name, value in stats.items():
        print(f"{name}: {value}")

    if do_save:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for entry in valid_results:
                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return stats


def main(args: argparse.Namespace) -> None:
    filter_agent_trajectories(args.result_path, args.do_save)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--do_save", action="store_true")
    main(parser.parse_args())
