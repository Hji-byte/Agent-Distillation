"""Filter prevalidated SFT rows by the exact tokenizer limits used for training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.repair.sft import _apply_template, _content_token_length  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_sft_by_token_limits(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer_path: str,
    max_length: int,
    max_assistant_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rows = [
        json.loads(line)
        for line in input_path.open(encoding="utf-8")
        if line.strip()
    ]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    kept_by_run_tag: Counter[str] = Counter()
    dropped_by_run_tag: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Invalid SFT row at {input_path}:{row_index}")
        sequence_length = len(_apply_template(tokenizer, messages, add_generation_prompt=False))
        assistant_lengths = [
            _content_token_length(tokenizer, message["content"])
            for message in messages
            if message.get("role") == "assistant"
        ]
        largest_assistant = max(assistant_lengths, default=0)
        metadata = row.get("metadata") or {}
        run_tag = str(metadata.get("run_tag") or "unknown")
        reasons = []
        if sequence_length > max_length:
            reasons.append("sequence_length")
        if largest_assistant > max_assistant_tokens:
            reasons.append("assistant_length")
        if reasons:
            dropped_by_run_tag[run_tag] += 1
            dropped.append(
                {
                    "row_index": row_index,
                    "repair_id": metadata.get("repair_id"),
                    "run_tag": run_tag,
                    "sequence_length": sequence_length,
                    "max_assistant_tokens": largest_assistant,
                    "reasons": reasons,
                }
            )
        else:
            kept_by_run_tag[run_tag] += 1
            kept.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in kept:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    summary = {
        "schema_version": "token-limit-filter-v1",
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "tokenizer_path": tokenizer_path,
        "max_length": max_length,
        "max_assistant_tokens": max_assistant_tokens,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "kept_by_run_tag": dict(sorted(kept_by_run_tag.items())),
        "dropped_by_run_tag": dict(sorted(dropped_by_run_tag.items())),
        "dropped": dropped,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=2048)
    args = parser.parse_args()
    summary = filter_sft_by_token_limits(
        args.input,
        args.output,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        max_assistant_tokens=args.max_assistant_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
