"""Split only over-length ALFWorld SFT trajectories at assistant-turn boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


INPUT_SCHEMA = "alfworld-teacher-sft-v1"
OUTPUT_SCHEMA = "alfworld-teacher-windowed-sft-v1"


def _token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(encoded, "keys"):
        return list(encoded["input_ids"])
    return list(encoded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_messages(messages: Any, source: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{source}: invalid messages")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError(f"{source}: expected system then task user")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"{source}: trajectory must end with assistant")
    for index, message in enumerate(messages[2:], start=2):
        expected = "assistant" if index % 2 == 0 else "user"
        if message.get("role") != expected:
            raise ValueError(
                f"{source}: message {index} has role {message.get('role')!r}, "
                f"expected {expected!r}"
            )
    return messages


def _segment_messages(
    messages: list[dict[str, str]],
    *,
    start_step: int,
    end_step: int,
) -> list[dict[str, str]]:
    task = dict(messages[1])
    if start_step > 0:
        boundary_observation = messages[2 * start_step + 1]
        task["content"] = (
            f"{task['content']}\n\n"
            "The interaction is already in progress. The latest environment feedback is:\n"
            f"{boundary_observation['content']}"
        )

    first_assistant = 2 + 2 * start_step
    last_assistant = 2 + 2 * (end_step - 1)
    return [dict(messages[0]), task] + [
        dict(message) for message in messages[first_assistant : last_assistant + 1]
    ]


def split_row(
    tokenizer,
    row: dict[str, Any],
    *,
    max_length: int,
    source_line: int,
) -> list[dict[str, Any]]:
    if row.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"line {source_line}: unsupported schema")
    if row.get("supervision") != "all_assistant_turns":
        raise ValueError(f"line {source_line}: unsupported supervision")
    messages = _validate_messages(row.get("messages"), f"line {source_line}")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"line {source_line}: missing metadata")

    assistant_count = sum(message["role"] == "assistant" for message in messages)
    if assistant_count != metadata.get("steps"):
        raise ValueError(
            f"line {source_line}: assistant turns {assistant_count} != steps {metadata.get('steps')}"
        )

    full_length = len(_token_ids(tokenizer, messages))
    ranges: list[tuple[int, int, int]] = []
    if full_length <= max_length:
        ranges.append((0, assistant_count, full_length))
    else:
        start = 0
        while start < assistant_count:
            best_end = None
            best_length = None
            for end in range(start + 1, assistant_count + 1):
                candidate = _segment_messages(messages, start_step=start, end_step=end)
                candidate_length = len(_token_ids(tokenizer, candidate))
                if candidate_length > max_length:
                    break
                best_end = end
                best_length = candidate_length
            if best_end is None or best_length is None:
                minimum = _segment_messages(messages, start_step=start, end_step=start + 1)
                minimum_length = len(_token_ids(tokenizer, minimum))
                raise ValueError(
                    f"line {source_line}: system, task, boundary observation, and step {start} "
                    f"already require {minimum_length} tokens (limit {max_length})"
                )
            ranges.append((start, best_end, best_length))
            start = best_end

    output: list[dict[str, Any]] = []
    for segment_index, (start, end, token_length) in enumerate(ranges):
        segment_messages = (
            [dict(message) for message in messages]
            if len(ranges) == 1
            else _segment_messages(messages, start_step=start, end_step=end)
        )
        output.append(
            {
                "schema_version": OUTPUT_SCHEMA,
                "messages": segment_messages,
                "supervision": "all_assistant_turns",
                "metadata": {
                    **metadata,
                    "source_trajectory_steps": assistant_count,
                    "source_token_length": full_length,
                    "segment_index": segment_index,
                    "segment_count": len(ranges),
                    "segment_start_step": start,
                    "segment_end_step_exclusive": end,
                    "steps": end - start,
                    "token_length": token_length,
                    "token_limit": max_length,
                    "boundary_observation_added": start > 0,
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=5600)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    output_rows = 0
    split_trajectories = 0
    input_actions = 0
    output_actions = 0
    token_lengths: list[int] = []
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            segments = split_row(
                tokenizer,
                row,
                max_length=args.max_length,
                source_line=line_number,
            )
            input_rows += 1
            input_actions += int(row["metadata"]["steps"])
            split_trajectories += len(segments) > 1
            for segment in segments:
                writer.write(json.dumps(segment, ensure_ascii=False) + "\n")
                output_rows += 1
                output_actions += int(segment["metadata"]["steps"])
                token_lengths.append(int(segment["metadata"]["token_length"]))

    if input_actions != output_actions:
        raise RuntimeError(
            f"Supervised action count changed: {input_actions} -> {output_actions}"
        )
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "input": args.input.as_posix(),
        "input_sha256": _sha256(source),
        "output": args.output.as_posix(),
        "output_sha256": _sha256(destination),
        "tokenizer_artifacts_sha256": {
            name: _sha256(Path(args.tokenizer).resolve() / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "max_length": args.max_length,
        "input_trajectories": input_rows,
        "output_segments": output_rows,
        "split_trajectories": split_trajectories,
        "unchanged_trajectories": input_rows - split_trajectories,
        "supervised_assistant_turns_before": input_actions,
        "supervised_assistant_turns_after": output_actions,
        "segment_token_length_min": min(token_lengths),
        "segment_token_length_max": max(token_lengths),
        "segment_token_length_mean": sum(token_lengths) / len(token_lengths),
    }
    summary_path = destination.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "summary_file": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
