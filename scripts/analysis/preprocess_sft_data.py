"""Materialize filtered agent trajectories as student-ready SFT messages."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "exps_research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "exps_research"))

from exps_research.train_utils.preprocess import preprocess_logs  # noqa: E402


def _sequence_length(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _max_assistant_turn_length(
    tokenizer,
    messages: list[dict[str, str]],
) -> int:
    """Return the largest number of target tokens in one assistant turn."""
    lengths = [
        len(tokenizer(message["content"], add_special_tokens=False)["input_ids"])
        for message in messages
        if message["role"] == "assistant"
    ]
    return max(lengths, default=0)


def materialize_sft_data(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    max_length: int,
    max_assistant_tokens: int,
) -> dict[str, object]:
    dataset = preprocess_logs(str(input_path), print_first=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    lengths: list[int] = []
    max_assistant_turn_lengths: list[int] = []
    kept_lengths: list[int] = []
    kept = 0
    dropped_over_max_length: list[int] = []
    dropped_over_max_assistant_tokens: list[int] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for row_index, row in enumerate(dataset):
            messages = row["messages"]
            sequence_length = _sequence_length(tokenizer, messages)
            max_assistant_turn_length = _max_assistant_turn_length(tokenizer, messages)
            lengths.append(sequence_length)
            max_assistant_turn_lengths.append(max_assistant_turn_length)
            if sequence_length > max_length:
                dropped_over_max_length.append(row_index)
            if max_assistant_turn_length > max_assistant_tokens:
                dropped_over_max_assistant_tokens.append(row_index)
            if (
                sequence_length > max_length
                or max_assistant_turn_length > max_assistant_tokens
            ):
                continue
            output_file.write(
                json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            )
            kept += 1
            kept_lengths.append(sequence_length)

    sorted_lengths = sorted(lengths)
    sorted_assistant_lengths = sorted(max_assistant_turn_lengths)
    p95_index = max(0, min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95) - 1))
    dropped_row_indices = sorted(
        set(dropped_over_max_length) | set(dropped_over_max_assistant_tokens)
    )
    stats: dict[str, object] = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "tokenizer_path": tokenizer_path,
        "max_length": max_length,
        "max_assistant_tokens": max_assistant_tokens,
        "total": len(lengths),
        "kept": kept,
        "dropped": len(dropped_row_indices),
        "dropped_over_max_length": len(dropped_over_max_length),
        "dropped_over_max_length_row_indices": dropped_over_max_length,
        "dropped_over_max_assistant_tokens": len(dropped_over_max_assistant_tokens),
        "dropped_over_max_assistant_tokens_row_indices": dropped_over_max_assistant_tokens,
        "dropped_row_indices": dropped_row_indices,
        "length_min": min(lengths) if lengths else 0,
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_p95": sorted_lengths[p95_index] if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "kept_length_max": max(kept_lengths) if kept_lengths else 0,
        "assistant_turn_length_median": (
            statistics.median(max_assistant_turn_lengths)
            if max_assistant_turn_lengths
            else 0
        ),
        "assistant_turn_length_p95": (
            sorted_assistant_lengths[p95_index] if sorted_assistant_lengths else 0
        ),
        "assistant_turn_length_max": (
            max(max_assistant_turn_lengths) if max_assistant_turn_lengths else 0
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats["summary_path"] = str(summary_path.resolve())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=2048)
    args = parser.parse_args()

    stats = materialize_sft_data(
        args.input,
        args.output,
        args.tokenizer,
        args.max_length,
        args.max_assistant_tokens,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
