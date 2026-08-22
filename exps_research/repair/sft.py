"""Materialization and supervision-aware loss masks for repair experiments."""

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


def _content_token_length(tokenizer, content: str) -> int:
    encoded = tokenizer(content, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return len(encoded)


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Tokenizer does not define the required ChatML token {token!r}")
    return int(token_id)


def _chatml_assistant_spans(
    tokenizer,
    full_ids: list[int],
    messages: list[dict[str, str]],
) -> dict[int, tuple[int, int]]:
    """Locate assistant bodies in one fully rendered ChatML conversation.

    Rendering message prefixes is not reliable for Qwen3.5: a turn rendered as
    the final assistant turn can receive empty ``<think>`` tags that are absent
    when the same turn is rendered inside the complete trajectory. Instead we
    render once and align each ChatML message header with the source role.
    """
    im_start_id = _token_id(tokenizer, "<|im_start|>")
    im_end_id = _token_id(tokenizer, "<|im_end|>")
    message_starts = [index for index, token_id in enumerate(full_ids) if token_id == im_start_id]
    if len(message_starts) != len(messages):
        raise ValueError(
            "ChatML message boundary count differs from the source trajectory; "
            "a message may contain an unescaped <|im_start|> token"
        )

    spans: dict[int, tuple[int, int]] = {}
    for message_index, (message, start) in enumerate(zip(messages, message_starts, strict=True)):
        role = message.get("role")
        role_ids = tokenizer.encode(str(role), add_special_tokens=False)
        if not role_ids:
            raise ValueError(f"Tokenizer produced no role tokens for message role {role!r}")
        role_start = start + 1
        if full_ids[role_start : role_start + len(role_ids)] != list(role_ids):
            raise ValueError(
                f"Rendered ChatML role at message {message_index} does not match {role!r}"
            )
        if role != "assistant":
            continue

        body_start = role_start + len(role_ids)
        # Qwen ChatML places a newline between the role header and content. It
        # belongs to the template, not the supervised assistant action.
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if newline_ids and full_ids[body_start : body_start + len(newline_ids)] == list(newline_ids):
            body_start += len(newline_ids)
        next_start = (
            message_starts[message_index + 1]
            if message_index + 1 < len(message_starts)
            else len(full_ids)
        )
        end_markers = [
            index for index in range(body_start, next_start) if full_ids[index] == im_end_id
        ]
        if len(end_markers) != 1:
            raise ValueError(
                f"Assistant message {message_index} must contain exactly one <|im_end|> boundary"
            )
        # Include <|im_end|> in the target, matching the original multi-turn
        # baseline collator and teaching the model when to end an action.
        spans[message_index] = (body_start, end_markers[0] + 1)
    return spans


def tokenize_last_assistant_only(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int = 4096,
    max_assistant_tokens: int = 2048,
) -> dict[str, Any]:
    """Tokenize a repair example and supervise only its final assistant turn."""
    return tokenize_supervised_messages(
        tokenizer,
        messages,
        supervision="last_assistant_only",
        max_length=max_length,
        max_assistant_tokens=max_assistant_tokens,
    )


def tokenize_all_assistant_turns(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int = 4096,
    max_assistant_tokens: int = 2048,
) -> dict[str, Any]:
    """Tokenize an ordinary teacher trajectory and supervise every assistant turn."""
    return tokenize_supervised_messages(
        tokenizer,
        messages,
        supervision="all_assistant_turns",
        max_length=max_length,
        max_assistant_tokens=max_assistant_tokens,
    )


def tokenize_supervised_messages(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    supervision: str,
    max_length: int = 4096,
    max_assistant_tokens: int = 2048,
) -> dict[str, Any]:
    """Tokenize one trajectory without silently truncating supervised content.

    ``all_assistant_turns`` reproduces ordinary Agent Distillation supervision:
    every teacher assistant action receives loss while system, user, and
    observation turns remain context only. ``last_assistant_only`` is used by
    local repair examples, where only the replacement action receives loss.
    """
    if supervision not in {"all_assistant_turns", "last_assistant_only"}:
        raise ValueError(f"Unsupported supervision mode: {supervision}")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with an assistant action")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_assistant_tokens <= 0:
        raise ValueError("max_assistant_tokens must be positive")

    full_ids = _apply_template(tokenizer, messages, add_generation_prompt=False)
    if len(full_ids) > max_length:
        raise ValueError(
            f"Tokenized sequence has {len(full_ids)} tokens, exceeding max_length={max_length}; "
            "refusing to train on a silently truncated trajectory"
        )

    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("SFT messages do not contain an assistant action")
    supervised_indices = (
        assistant_indices if supervision == "all_assistant_turns" else assistant_indices[-1:]
    )
    assistant_spans = _chatml_assistant_spans(tokenizer, full_ids, messages)
    if set(assistant_spans) != set(assistant_indices):
        raise ValueError("Rendered assistant boundaries do not match the source trajectory")

    labels = [-100] * len(full_ids)
    supervised_lengths: list[int] = []
    for message_index in supervised_indices:
        target_start, target_end = assistant_spans[message_index]
        labeled_length = target_end - target_start
        if labeled_length <= 0:
            raise ValueError("Assistant target is empty after applying the chat template")
        content_length = _content_token_length(tokenizer, messages[message_index]["content"])
        if content_length > max_assistant_tokens:
            raise ValueError(
                f"Assistant content has {content_length} tokens, exceeding "
                f"max_assistant_tokens={max_assistant_tokens}"
            )
        labels[target_start:target_end] = full_ids[target_start:target_end]
        supervised_lengths.append(labeled_length)

    if not any(label != -100 for label in labels):
        raise ValueError("No assistant tokens were selected for supervision")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sequence_length": len(full_ids),
        "supervised_token_count": sum(supervised_lengths),
    }


__all__ = [
    "accepted_repair_example",
    "materialize_accepted_repairs",
    "materialize_repair_jsonl",
    "tokenize_all_assistant_turns",
    "tokenize_last_assistant_only",
    "tokenize_supervised_messages",
]
