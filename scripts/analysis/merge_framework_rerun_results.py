#!/usr/bin/env python3
"""Replace framework-lost Math500 records with their isolated rerun results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KNOWN_FRAMEWORK_ERROR = "string indices must be integers, not 'str'"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Record {line_number} in {path} is not an object")
            if not isinstance(record.get("question"), str):
                raise ValueError(f"Record {line_number} in {path} has no question")
            records.append(record)
    return records


def _is_replaceable_framework_error(record: dict[str, Any]) -> bool:
    return (
        record.get("generated_answer") is None
        and record.get("log_data") is None
        and KNOWN_FRAMEWORK_ERROR in str(record.get("error") or "")
    )


def merge_framework_rerun(
    baseline_path: Path,
    rerun_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    baseline = _load_jsonl(baseline_path)
    rerun = _load_jsonl(rerun_path)

    baseline_questions = [record["question"] for record in baseline]
    if len(set(baseline_questions)) != len(baseline_questions):
        raise ValueError("Baseline contains duplicate questions")

    replaceable = {
        record["question"]
        for record in baseline
        if _is_replaceable_framework_error(record)
    }
    rerun_by_question = {record["question"]: record for record in rerun}
    if len(rerun_by_question) != len(rerun):
        raise ValueError("Framework rerun contains duplicate questions")
    if set(rerun_by_question) != replaceable:
        missing = sorted(replaceable - set(rerun_by_question))
        extra = sorted(set(rerun_by_question) - replaceable)
        raise ValueError(
            "Rerun questions do not exactly match replaceable framework errors: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    for question, record in rerun_by_question.items():
        if (
            record.get("generated_answer") is None
            or not isinstance(record.get("log_data"), dict)
            or record.get("error")
        ):
            raise ValueError(f"Rerun did not produce a valid recorded attempt: {question}")

    merged = [rerun_by_question.get(record["question"], record) for record in baseline]
    correct_answers = sum(bool(record.get("score")) for record in merged)
    remaining_framework_errors = sum(
        _is_replaceable_framework_error(record) for record in merged
    )
    summary = {
        "baseline_file": baseline_path.as_posix(),
        "framework_rerun_file": rerun_path.as_posix(),
        "output_file": output_path.as_posix(),
        "raw_records": len(merged),
        "duplicate_records_ignored": 0,
        "replaced_framework_errors": len(replaceable),
        "remaining_framework_errors": remaining_framework_errors,
        "total_questions": len(merged),
        "correct_answers": correct_answers,
        "accuracy": correct_answers / len(merged) if merged else 0.0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--framework-rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    summary = merge_framework_rerun(
        args.baseline,
        args.framework_rerun,
        args.output,
        args.summary,
    )
    print(
        f"Merged {summary['replaced_framework_errors']} framework reruns: "
        f"{summary['correct_answers']}/{summary['total_questions']} "
        f"({summary['accuracy']:.2%})"
    )


if __name__ == "__main__":
    main()
