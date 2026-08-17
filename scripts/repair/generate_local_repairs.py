"""Generate verified one-step repairs from scored failed student trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.repair import RepairConfig, RepairPipeline
from exps_research.unified_framework.models import setup_model


def repair_id(entry: dict) -> str:
    task_id = (((entry.get("log_data") or {}).get("metadata") or {}).get("task_id"))
    if task_id is not None:
        return str(task_id)
    return hashlib.sha256(str(entry.get("question", "")).encode("utf-8")).hexdigest()[:16]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                completed.add(str(json.loads(line)["repair_id"]))
    return completed


def main(args) -> None:
    source = Path(args.input)
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Repair output must differ from the source trajectory file")
    output.parent.mkdir(parents=True, exist_ok=True)

    teacher = setup_model(
        model_type="openai",
        model_id=args.teacher_model_id,
        max_tokens=args.teacher_max_tokens,
        temperature=0.0,
        seed=args.seed,
    )
    if args.continuation_model_type == "teacher":
        continuation = teacher
    else:
        if not args.continuation_model_id:
            raise ValueError("--continuation_model_id is required unless continuation_model_type=teacher")
        continuation = setup_model(
            model_type=args.continuation_model_type,
            model_id=args.continuation_model_id,
            fine_tuned=bool(args.continuation_lora_path),
            lora_path=args.continuation_lora_path,
            max_tokens=args.continuation_max_tokens,
            temperature=0.0,
            seed=args.seed,
        )

    pipeline = RepairPipeline(
        teacher_model=teacher,
        continuation_model=continuation,
        config=RepairConfig(
            max_candidates=args.max_candidates,
            max_continuation_steps=args.max_continuation_steps,
            max_format_retries=args.max_format_retries,
            execution_timeout_seconds=args.execution_timeout_seconds,
        ),
    )
    completed = load_completed(output) if args.resume else set()
    processed = accepted = rejected = 0
    seen_this_run = set()

    with source.open(encoding="utf-8") as reader, output.open("a", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("score") not in (0, False) or not entry.get("log_data"):
                continue
            current_id = repair_id(entry)
            if current_id in completed or current_id in seen_this_run:
                continue
            if args.max_entries > 0 and processed >= args.max_entries:
                break
            seen_this_run.add(current_id)
            try:
                outcome = pipeline.repair_entry(entry)
            except Exception as error:
                outcome = {
                    "schema_version": "local-repair-v1",
                    "repair_id": current_id,
                    "question": entry.get("question"),
                    "accepted": False,
                    "rejection_reason": f"pipeline_error: {type(error).__name__}: {error}",
                }
            writer.write(json.dumps(outcome, ensure_ascii=False, default=str) + "\n")
            writer.flush()
            processed += 1
            if outcome.get("accepted"):
                accepted += 1
            else:
                rejected += 1
            print(
                f"processed={processed} accepted={accepted} rejected={rejected} "
                f"repair_id={current_id}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Scored failed student trajectory JSONL")
    parser.add_argument("--output", required=True, help="Independent repair-attempt JSONL")
    parser.add_argument("--teacher_model_id", default="qwen3.5-27b")
    parser.add_argument("--teacher_max_tokens", type=int, default=1280)
    parser.add_argument(
        "--continuation_model_type",
        choices=["teacher", "transformers", "openai"],
        default="teacher",
        help="Use 'transformers' plus the trained S0 model for the research setting.",
    )
    parser.add_argument("--continuation_model_id")
    parser.add_argument("--continuation_lora_path")
    parser.add_argument("--continuation_max_tokens", type=int, default=1024)
    parser.add_argument("--max_candidates", type=int, default=5)
    parser.add_argument("--max_continuation_steps", type=int, default=4)
    parser.add_argument("--max_format_retries", type=int, default=1)
    parser.add_argument("--execution_timeout_seconds", type=int, default=30)
    parser.add_argument("--max_entries", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
