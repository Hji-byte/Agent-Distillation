"""Reformat ETO ALFWorld trajectories without modifying the raw outputs.

The ETO conversation-form prompt stores the instruction, a synthetic ``OK``,
and two ReAct demonstrations as separate chat turns.  This script materializes
an equivalent, training-friendly representation:

* system: instruction + the two rendered ReAct demonstrations
* user: ``Now, it's your turn ...`` + the current ALFWorld task
* remaining turns: the teacher's actions and environment observations

The raw source files are read-only.  Every source record is validated against
the pinned ReAct demonstrations before a derived record is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps_research.alfworld_eto.react_prompting import load_react_two_shot


ROLE_MAP = {"human": "user", "gpt": "assistant"}
TASK_HEADER = "Now, it's your turn and here is the task."


def _render_examples(demos: list[list[dict[str, str]]]) -> str:
    rendered = ""
    for demo_index, demo in enumerate(demos, start=1):
        rendered += f"Example task {demo_index}:\n"
        for message_index, message in enumerate(demo):
            content = message["content"]
            if message_index == 0:
                if message["role"] != "user":
                    raise ValueError("A ReAct demonstration must start with a user task")
                rendered += content + "\n"
            elif message["role"] == "assistant":
                rendered += content + "\n"
            elif message["role"] == "user":
                rendered += content + "\n\n"
            else:
                raise ValueError(f"Unsupported demonstration role: {message['role']}")
        if demo_index != len(demos):
            rendered += "\n"
    return rendered.rstrip()


def _to_raw_conversation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    reverse_roles = {"user": "human", "assistant": "gpt"}
    return [
        {"from": reverse_roles[message["role"]], "value": message["content"]}
        for message in messages
    ]


def reformat_record(
    record: dict[str, Any], *, eto_root: Path, source_file: str | None = None
) -> dict[str, Any]:
    record_label = source_file or str(record.get("meta", {}).get("error", "record"))
    meta = record.get("meta")
    conversations = record.get("conversations")
    if not isinstance(meta, dict) or not isinstance(conversations, list):
        raise ValueError(f"{record_label}: missing meta or conversations")

    steps = meta.get("steps")
    game_file = meta.get("error")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"{record_label}: invalid step count {steps!r}")
    if not isinstance(game_file, str) or not game_file:
        raise ValueError(f"{record_label}: missing ALFWorld game path")

    current_length = 1 + 2 * steps
    current_start = len(conversations) - current_length
    if current_start < 2:
        raise ValueError(f"{record_label}: conversation is too short for {steps} steps")
    if conversations[0].get("from") != "human":
        raise ValueError(f"{record_label}: first message is not the instruction")
    if conversations[1] != {"from": "gpt", "value": "OK"}:
        raise ValueError(f"{record_label}: expected ETO's fixed assistant OK")

    prompt_type, demos, demo_keys = load_react_two_shot(eto_root, game_file)
    expected_demo_turns = _to_raw_conversation(
        [message for demo in demos for message in demo]
    )
    actual_demo_turns = conversations[2:current_start]
    if actual_demo_turns != expected_demo_turns:
        raise ValueError(
            f"{record_label}: saved ICL turns do not match {demo_keys}"
        )

    current = conversations[current_start:]
    if current[0].get("from") != "human":
        raise ValueError(f"{record_label}: current task does not start with human")
    for index, message in enumerate(current):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if message.get("from") != expected_role:
            raise ValueError(
                f"{record_label}: current turn {index} is {message.get('from')}, "
                f"expected {expected_role}"
            )

    instruction = conversations[0]["value"]
    examples = _render_examples(demos)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{instruction}\n---\nHere are 2 examples.\n\n{examples}",
        },
        {
            "role": "user",
            "content": f"{TASK_HEADER}\n{current[0]['value']}",
        },
    ]
    messages.extend(
        {"role": ROLE_MAP[message["from"]], "content": message["value"]}
        for message in current[1:]
    )

    metadata = {
        **meta,
        "prompt_profile": "react-type-2shot",
        "prompt_type": prompt_type,
        "react_example_keys": demo_keys,
        "raw_prompt_turns": current_start,
    }
    if source_file is not None:
        metadata["source_file"] = source_file

    return {
        "schema_version": "alfworld-eto-react2-system-user-v1",
        "messages": messages,
        "supervision": "all_assistant_turns",
        "metadata": metadata,
    }


def _numeric_json_files(directory: Path) -> list[Path]:
    paths = [path for path in directory.glob("*.json") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eto-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    eto_root = args.eto_root.resolve()
    if source == destination:
        raise ValueError("Input and output directories must be different")
    if not source.is_dir():
        raise FileNotFoundError(source)

    files = _numeric_json_files(source)
    if not files:
        raise ValueError(f"No numeric JSON trajectory files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    total_teacher_steps = 0
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        normalized = reformat_record(
            record, eto_root=eto_root, source_file=path.name
        )
        output_path = destination / path.name
        output_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        is_success = normalized["metadata"]["success"] is True
        success += int(is_success)
        failed += int(not is_success)
        total_teacher_steps += normalized["metadata"]["steps"]

    summary = {
        "schema_version": "alfworld-eto-react2-system-user-v1",
        "source_directory": str(source),
        "output_directory": str(destination),
        "source_files_modified": 0,
        "records": len(files),
        "success": success,
        "failed": failed,
        "teacher_steps": total_teacher_steps,
        "system_messages": len(files),
        "fixed_ok_messages_retained": 0,
        "separate_icl_chat_turns_retained": 0,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
