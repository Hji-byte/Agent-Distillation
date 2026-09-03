"""Build successful ALFWorld teacher SFT rows from complete trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "alfworld-eto-react2-system-user-v1"
OUTPUT_SCHEMA = "alfworld-teacher-sft-v1"


def build_sft_row(record: dict[str, Any], *, source_file: str) -> dict[str, Any] | None:
    if record.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"{source_file}: unsupported trajectory schema")
    metadata = record.get("metadata")
    messages = record.get("messages")
    if not isinstance(metadata, dict) or not isinstance(messages, list):
        raise ValueError(f"{source_file}: missing metadata or messages")
    if metadata.get("success") is not True:
        return None
    if len(messages) < 4:
        raise ValueError(f"{source_file}: successful trajectory is too short")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError(f"{source_file}: expected system then current-task user")

    teacher_steps = metadata.get("steps")
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    if assistant_count != teacher_steps:
        raise ValueError(
            f"{source_file}: assistant turns {assistant_count} != steps {teacher_steps}"
        )

    terminal = messages[-1]
    if terminal.get("role") != "user" or not str(terminal.get("content", "")).startswith(
        "Observation:"
    ):
        raise ValueError(f"{source_file}: missing terminal environment observation")

    training_messages = [dict(message) for message in messages[:-1]]
    if training_messages[-1].get("role") != "assistant":
        raise ValueError(f"{source_file}: SFT row does not end with assistant")
    if sum(message["role"] == "assistant" for message in training_messages) != teacher_steps:
        raise ValueError(f"{source_file}: removing terminal observation lost an action")

    return {
        "schema_version": OUTPUT_SCHEMA,
        "messages": training_messages,
        "supervision": "all_assistant_turns",
        "metadata": {
            **metadata,
            "source_file": source_file,
            "verified_success": True,
            "terminal_success_observation_removed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input_dir.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing SFT file: {output}")

    files = sorted(
        (path for path in source.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not files:
        raise ValueError(f"No numeric trajectory files found in {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    excluded_failed = 0
    assistant_turns = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            row = build_sft_row(record, source_file=path.name)
            if row is None:
                excluded_failed += 1
                continue
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            assistant_turns += int(row["metadata"]["steps"])

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "input_directory": str(source),
        "output_file": str(output),
        "complete_trajectories_modified": 0,
        "input_records": len(files),
        "successful_sft_rows": kept,
        "excluded_failed_rows": excluded_failed,
        "terminal_observations_removed": kept,
        "supervised_assistant_turns": assistant_turns,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "summary_file": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
