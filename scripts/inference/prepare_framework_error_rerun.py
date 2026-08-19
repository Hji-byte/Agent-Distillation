#!/usr/bin/env python3
"""Build a dataset containing only benchmark records lost to framework errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KNOWN_FRAMEWORK_ERROR = "string indices must be integers, not 'str'"


def _load_examples(dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Dataset has no examples list: {dataset_path}")
    return payload.get("metadata") or {}, examples


def _framework_error_questions(result_path: Path) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    with result_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            error = str(record.get("error") or "")
            is_framework_error = (
                record.get("generated_answer") is None
                and record.get("log_data") is None
                and KNOWN_FRAMEWORK_ERROR in error
            )
            if not is_framework_error:
                continue
            question = record.get("question")
            if not isinstance(question, str) or not question:
                raise ValueError(
                    f"Framework-error record at line {line_number} has no question"
                )
            if question not in seen:
                seen.add(question)
                questions.append(question)
    return questions


def prepare_rerun_dataset(
    dataset_path: Path,
    result_path: Path,
    output_path: Path,
) -> int:
    metadata, examples = _load_examples(dataset_path)
    questions = _framework_error_questions(result_path)
    if not questions:
        raise ValueError(
            "No matching framework-error records were found; refusing to create an empty rerun"
        )

    by_question = {example.get("question"): example for example in examples}
    missing = [question for question in questions if question not in by_question]
    if missing:
        raise ValueError(f"{len(missing)} failed questions were not found in the source dataset")

    rerun_examples = [by_question[question] for question in questions]
    rerun_metadata = dict(metadata)
    rerun_metadata.update(
        {
            "repair_kind": "framework_error_only",
            "source_dataset": str(dataset_path),
            "source_result": str(result_path),
            "framework_error": KNOWN_FRAMEWORK_ERROR,
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
    return len(rerun_examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = prepare_rerun_dataset(args.dataset, args.results, args.output)
    print(f"Prepared {count} framework-error questions: {args.output}")


if __name__ == "__main__":
    main()
