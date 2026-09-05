#!/usr/bin/env python3
"""Prepare and merge a higher-token rerun of truncated CoT records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Dataset has no examples list: {path}")
    _validate_unique_questions(examples, f"dataset {path}")
    return payload.get("metadata") or {}, examples


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Record {line_number} in {path} is not an object")
            if not isinstance(record.get("question"), str) or not record["question"]:
                raise ValueError(f"Record {line_number} in {path} has no question")
            records.append(record)
    _validate_unique_questions(records, f"result file {path}")
    return records


def _validate_unique_questions(records: list[dict[str, Any]], label: str) -> None:
    questions = [record.get("question") for record in records]
    if any(not isinstance(question, str) or not question for question in questions):
        raise ValueError(f"Every record in {label} must have a non-empty question")
    if len(set(questions)) != len(questions):
        raise ValueError(f"{label} contains duplicate questions")


def _validate_complete_baseline(
    examples: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> None:
    dataset_questions = {example["question"] for example in examples}
    baseline_questions = {record["question"] for record in baseline}
    missing = dataset_questions - baseline_questions
    extra = baseline_questions - dataset_questions
    if missing or extra:
        raise ValueError(
            "Baseline does not exactly cover the source dataset: "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def _truncated_questions(
    baseline: list[dict[str, Any]], source_max_tokens: int
) -> set[str]:
    truncated: set[str] = set()
    for record in baseline:
        output_tokens = record.get("output_tokens")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            raise ValueError(
                "Every baseline record must contain an integer output_tokens value"
            )
        if output_tokens >= source_max_tokens:
            truncated.add(record["question"])
    if not truncated:
        raise ValueError(
            f"No records reached the {source_max_tokens}-token source limit"
        )
    return truncated


def prepare_truncated_dataset(
    dataset_path: Path,
    baseline_path: Path,
    output_path: Path,
    source_max_tokens: int,
    rerun_max_tokens: int,
) -> dict[str, Any]:
    if rerun_max_tokens <= source_max_tokens:
        raise ValueError("rerun_max_tokens must be greater than source_max_tokens")

    metadata, examples = _load_dataset(dataset_path)
    baseline = _load_jsonl(baseline_path)
    _validate_complete_baseline(examples, baseline)
    truncated = _truncated_questions(baseline, source_max_tokens)
    rerun_examples = [
        example for example in examples if example["question"] in truncated
    ]

    rerun_metadata = dict(metadata)
    rerun_metadata.update(
        {
            "subset_kind": "cot_hard_truncation_rerun",
            "source_dataset": str(dataset_path),
            "source_baseline": str(baseline_path),
            "source_max_tokens": source_max_tokens,
            "rerun_max_tokens": rerun_max_tokens,
            "num_examples": len(rerun_examples),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"metadata": rerun_metadata, "examples": rerun_examples},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_questions": len(examples),
        "truncated_questions": len(rerun_examples),
        "source_max_tokens": source_max_tokens,
        "rerun_max_tokens": rerun_max_tokens,
        "output": str(output_path),
    }


def merge_truncated_rerun(
    dataset_path: Path,
    baseline_path: Path,
    rerun_path: Path,
    output_path: Path,
    summary_path: Path,
    source_max_tokens: int,
    rerun_max_tokens: int,
) -> dict[str, Any]:
    if rerun_max_tokens <= source_max_tokens:
        raise ValueError("rerun_max_tokens must be greater than source_max_tokens")

    _, examples = _load_dataset(dataset_path)
    baseline = _load_jsonl(baseline_path)
    rerun = _load_jsonl(rerun_path)
    _validate_complete_baseline(examples, baseline)

    truncated = _truncated_questions(baseline, source_max_tokens)
    rerun_by_question = {record["question"]: record for record in rerun}
    missing = truncated - set(rerun_by_question)
    extra = set(rerun_by_question) - truncated
    if missing or extra:
        raise ValueError(
            "Rerun does not exactly cover the truncated baseline questions: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    for question, record in rerun_by_question.items():
        if record.get("error") or record.get("generated_answer") is None:
            raise ValueError(f"Rerun did not produce a valid answer: {question}")
        output_tokens = record.get("output_tokens")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
            raise ValueError(f"Rerun record has invalid output_tokens: {question}")
        if output_tokens > rerun_max_tokens:
            raise ValueError(
                f"Rerun record exceeds the declared {rerun_max_tokens}-token limit: "
                f"{question}"
            )

    baseline_by_question = {record["question"]: record for record in baseline}
    merged = [
        rerun_by_question.get(
            example["question"], baseline_by_question[example["question"]]
        )
        for example in examples
    ]
    still_at_rerun_limit = sum(
        int(record.get("output_tokens", 0) >= rerun_max_tokens)
        for record in rerun
    )
    summary = {
        "source_dataset": str(dataset_path),
        "baseline_file": str(baseline_path),
        "rerun_file": str(rerun_path),
        "output_file": str(output_path),
        "total_questions": len(merged),
        "source_max_tokens": source_max_tokens,
        "rerun_max_tokens": rerun_max_tokens,
        "replaced_truncated_questions": len(truncated),
        "retained_baseline_questions": len(merged) - len(truncated),
        "reruns_reaching_new_limit": still_at_rerun_limit,
        "merged_output_tokens": sum(
            int(record.get("output_tokens", 0)) for record in merged
        ),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Create a dataset containing only source-limit records"
    )
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--source-max-tokens", type=int, default=4096)
    prepare.add_argument("--rerun-max-tokens", type=int, default=8192)

    merge = subparsers.add_parser(
        "merge", help="Replace source-limit records with completed reruns"
    )
    merge.add_argument("--dataset", type=Path, required=True)
    merge.add_argument("--baseline", type=Path, required=True)
    merge.add_argument("--rerun", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--summary", type=Path, required=True)
    merge.add_argument("--source-max-tokens", type=int, default=4096)
    merge.add_argument("--rerun-max-tokens", type=int, default=8192)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        summary = prepare_truncated_dataset(
            args.dataset,
            args.baseline,
            args.output,
            args.source_max_tokens,
            args.rerun_max_tokens,
        )
        print(
            f"Prepared {summary['truncated_questions']}/"
            f"{summary['source_questions']} truncated questions: {args.output}"
        )
        return

    summary = merge_truncated_rerun(
        args.dataset,
        args.baseline,
        args.rerun,
        args.output,
        args.summary,
        args.source_max_tokens,
        args.rerun_max_tokens,
    )
    print(
        f"Replaced {summary['replaced_truncated_questions']} truncated records; "
        f"merged {summary['total_questions']} questions: {args.output}"
    )


if __name__ == "__main__":
    main()
