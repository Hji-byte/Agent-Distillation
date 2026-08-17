"""Merge prevalidated SFT JSONL files while deduplicating by task text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Invalid SFT row at {path}:{line_number}")
            rows.append(row)
    return rows


def _task_key(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                break
            return content.strip()
    raise ValueError("SFT row does not contain a non-empty user task message")


def merge_sft_files(
    sources: list[tuple[str, Path]],
    output_path: Path,
    *,
    max_length: int,
    max_assistant_tokens: int,
    tokenizer_path: str,
) -> dict[str, Any]:
    seen_tasks: set[str] = set()
    merged_rows: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []

    for label, path in sources:
        rows = _load_rows(path)
        kept = 0
        duplicates = 0
        for row in rows:
            task = _task_key(row)
            if task in seen_tasks:
                duplicates += 1
                continue
            seen_tasks.add(task)
            merged_rows.append(row)
            kept += 1
        source_stats.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "input_rows": len(rows),
                "kept_rows": kept,
                "duplicate_tasks": duplicates,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in merged_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    summary = {
        "schema_version": "merged-sft-v1",
        "output_path": str(output_path.resolve()),
        "order": [label for label, _ in sources],
        "sources": source_stats,
        "total_input_rows": sum(item["input_rows"] for item in source_stats),
        "total_duplicate_tasks": sum(item["duplicate_tasks"] for item in source_stats),
        "total_kept": len(merged_rows),
        "max_length": max_length,
        "max_assistant_tokens": max_assistant_tokens,
        "tokenizer_path": tokenizer_path,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    summary = merge_sft_files(
        [(label, Path(path)) for label, path in args.source],
        args.output,
        max_length=args.max_length,
        max_assistant_tokens=args.max_assistant_tokens,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
