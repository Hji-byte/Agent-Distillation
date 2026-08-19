"""Materialization and last-assistant-only loss masks for repair SFT."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def accepted_repair_example(outcome: dict[str, Any]) -> dict[str, Any] | None:
    if not outcome.get("accepted"):
        return None
    attempts = outcome.get("attempts") or []
    index = int(outcome["selected_attempt_index"])
    attempt = attempts[index]
    return {
        "schema_version": "local-repair-sft-v1",
        "messages": attempt["sft_messages"],
        "supervision": "last_assistant_only",
        "target_assistant_turn_index": attempt["target_assistant_turn_index"],
        "metadata": {
            "repair_id": outcome["repair_id"],
            "failure_kind": outcome["failure_kind"],
            "repair_step_index": outcome["selected_step_index"],
            "teacher_model_id": outcome["teacher_model_id"],
            "continuation_model_id": outcome["continuation_model_id"],
            "verification_mode": outcome.get("verification_mode"),
            "continuation_step_count": outcome.get("continuation_step_count"),
            "original_step_is_final": attempt.get("original_step_is_final"),
            "repaired_step_is_final": attempt.get("repaired_step_is_final"),
            "run_tag": (outcome.get("experiment_config") or {}).get("run_tag"),
        },
    }


def materialize_accepted_repairs(outcomes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [example for outcome in outcomes if (example := accepted_repair_example(outcome)) is not None]


def materialize_repair_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            example = accepted_repair_example(json.loads(line))
            if example is None:
                continue
            writer.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def _apply_template(tokenizer, messages, *, add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": None,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    # Transformers 5 returns BatchEncoding for Qwen3.5, while older/token-only
    # templates return the input-id list directly.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return list(encoded)


def tokenize_last_assistant_only(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    """Tokenize a repair example and mask every token before the final target.

    Earlier student assistant actions remain visible context but receive label
    ``-100``. Left truncation preserves the repaired target at sequence end.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Repair SFT messages must end with the repaired assistant action")
    full_ids = _apply_template(tokenizer, messages, add_generation_prompt=False)
    prompt_ids = _apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the prompt as a prefix of the full repair sample")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(full_ids) > max_length:
        offset = len(full_ids) - max_length
        full_ids = full_ids[offset:]
        labels = labels[offset:]
    if not any(label != -100 for label in labels):
        raise ValueError("Repair target was fully truncated")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


__all__ = [
    "accepted_repair_example",
    "materialize_accepted_repairs",
    "materialize_repair_jsonl",
    "tokenize_last_assistant_only",
]
