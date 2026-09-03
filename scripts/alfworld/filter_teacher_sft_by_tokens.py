"""Keep only complete ALFWorld SFT trajectories within an exact token limit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


INPUT_SCHEMA = "alfworld-teacher-sft-v1"
SUMMARY_SCHEMA = "alfworld-teacher-length-filter-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_length(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if hasattr(encoded, "keys"):
        return len(encoded["input_ids"])
    return len(encoded)


def _validate_row(row: dict[str, Any], line_number: int) -> None:
    if row.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"line {line_number}: unsupported schema")
    if row.get("supervision") != "all_assistant_turns":
        raise ValueError(f"line {line_number}: unsupported supervision")
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: missing messages or metadata")
    if not messages or messages[0].get("role") != "system":
        raise ValueError(f"line {line_number}: missing system prompt")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"line {line_number}: trajectory must end with assistant")
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    if assistant_count != metadata.get("steps"):
        raise ValueError(
            f"line {line_number}: assistant turns {assistant_count} != steps {metadata.get('steps')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=6400)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    tokenizer_root = Path(args.tokenizer).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[int, int, dict[str, Any]]] = []
    removed: list[dict[str, Any]] = []
    total_actions = 0
    with source.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _validate_row(row, line_number)
            token_length = _token_length(tokenizer, row["messages"])
            actions = int(row["metadata"]["steps"])
            total_actions += actions
            if token_length <= args.max_length:
                kept.append((line_number, token_length, row))
            else:
                removed.append(
                    {
                        "source_line": line_number,
                        "source_file": row["metadata"].get("source_file"),
                        "prompt_type": row["metadata"].get("prompt_type"),
                        "steps": actions,
                        "token_length": token_length,
                    }
                )

    with destination.open("w", encoding="utf-8", newline="\n") as writer:
        for _, _, row in kept:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept_actions = sum(int(row["metadata"]["steps"]) for _, _, row in kept)
    kept_lengths = [token_length for _, token_length, _ in kept]
    longest_index = max(range(len(kept)), key=lambda index: kept[index][1])
    tokenizer_hashes = {
        name: _sha256(tokenizer_root / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "input": args.input.as_posix(),
        "input_sha256": _sha256(source),
        "output": args.output.as_posix(),
        "output_sha256": _sha256(destination),
        "tokenizer_artifacts_sha256": tokenizer_hashes,
        "max_length": args.max_length,
        "input_trajectories": len(kept) + len(removed),
        "kept_complete_trajectories": len(kept),
        "removed_over_length_trajectories": len(removed),
        "supervised_assistant_turns_before": total_actions,
        "supervised_assistant_turns_after": kept_actions,
        "removed_supervised_assistant_turns": total_actions - kept_actions,
        "kept_token_length_min": min(kept_lengths),
        "kept_token_length_max": max(kept_lengths),
        "kept_token_length_mean": sum(kept_lengths) / len(kept_lengths),
        "longest_kept_output_zero_based_index": longest_index,
        "longest_kept_source_file": kept[longest_index][2]["metadata"].get("source_file"),
        "removed": removed,
    }
    summary_path = destination.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "removed"}
            | {"summary_file": str(summary_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
