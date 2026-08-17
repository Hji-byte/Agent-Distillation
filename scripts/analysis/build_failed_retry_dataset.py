"""Build retry datasets from questions without a valid scored trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_valid_questions(filtered_path: Path) -> set[str]:
    questions: set[str] = set()
    with filtered_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            if isinstance(question, str):
                questions.add(question)
    return questions


def build_retry_dataset(source_path: Path, filtered_path: Path, output_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    examples = source.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Dataset does not contain an examples list: {source_path}")

    valid_questions = load_valid_questions(filtered_path)
    retry_examples = [
        example
        for example in examples
        if example.get("question") not in valid_questions
    ]
    output = dict(source)
    output["examples"] = retry_examples
    output["retry_metadata"] = {
        "source_path": str(source_path.resolve()),
        "filtered_path": str(filtered_path.resolve()),
        "selection": "question_not_in_valid_scored_trajectory_set",
        "source_examples": len(examples),
        "valid_questions": len(valid_questions),
        "retry_examples": len(retry_examples),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output["retry_metadata"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = build_retry_dataset(args.source, args.filtered, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
