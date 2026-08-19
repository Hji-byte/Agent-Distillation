"""Generate verified one-step repairs from scored failed student trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.repair import RepairConfig, RepairPipeline
from exps_research.repair.messages import REPAIR_PROMPT_VERSION, repair_prompt_sha256
from exps_research.unified_framework.models import setup_model


def repair_id(entry: dict) -> str:
    task_id = (((entry.get("log_data") or {}).get("metadata") or {}).get("task_id"))
    if task_id is not None:
        return str(task_id)
    return hashlib.sha256(str(entry.get("question", "")).encode("utf-8")).hexdigest()[:16]


def truncate_incomplete_jsonl_tail(path: Path) -> bool:
    """Discard only an unterminated final record left by an interrupted write."""
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    final_newline = data.rfind(b"\n")
    path.write_bytes(data[: final_newline + 1] if final_newline >= 0 else b"")
    return True


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                entry = json.loads(line)
                if not entry.get("retryable_error"):
                    completed.add(str(entry["repair_id"]))
    return completed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_file_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def git_provenance() -> dict:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_files_dirty": bool(status) if status is not None else None,
    }


def runtime_provenance() -> dict:
    package_names = ("torch", "transformers", "peft", "smolagents", "openai")
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def build_experiment_config(args, source: Path) -> dict:
    dataset = Path(args.dataset_path) if args.dataset_path else None
    model_path = Path(args.continuation_model_id) if args.continuation_model_id else None
    adapter_path = Path(args.continuation_lora_path) if args.continuation_lora_path else None
    adapter_weights = None
    if adapter_path:
        adapter_weights = next(
            (
                path
                for path in (
                    adapter_path / "adapter_model.safetensors",
                    adapter_path / "adapter_model.bin",
                )
                if path.is_file()
            ),
            None,
        )
    return {
        "run_tag": args.run_tag,
        "code": git_provenance(),
        "runtime": runtime_provenance(),
        "teacher": {
            "model_type": "openai",
            "model_id": args.teacher_model_id,
            "max_tokens": args.teacher_max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "n": 1,
        },
        "continuation": {
            "model_type": args.continuation_model_type,
            "model_id": args.continuation_model_id,
            "lora_path": args.continuation_lora_path,
            "max_tokens": args.continuation_max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "n": 1,
        },
        "student_evaluation": {
            "model_id": args.continuation_model_id,
            "lora_path": args.continuation_lora_path,
            "max_tokens": args.student_evaluation_max_tokens,
            "max_steps": args.student_evaluation_max_steps,
            "temperature": 0.0,
            "seed": args.seed,
            "n": 1,
            "max_samples": args.student_evaluation_max_samples,
            "protocol": "agent_steps=5_code_only_v126_sft_qlora",
        },
        "repair": {
            "max_candidates": args.max_candidates,
            "max_continuation_steps": args.max_continuation_steps,
            "max_format_retries": args.max_format_retries,
            "execution_timeout_seconds": args.execution_timeout_seconds,
            "max_entries": args.max_entries,
        },
        "prompt": {
            "version": REPAIR_PROMPT_VERSION,
            "mode": "trajectory_prefix_only",
            "additional_instruction": None,
            "sha256": repair_prompt_sha256(),
        },
        "inputs": {
            "scored_trajectories_path": str(source.resolve()),
            "scored_trajectories_sha256": file_sha256(source),
            "dataset_path": str(dataset.resolve()) if dataset else None,
            "dataset_sha256": file_sha256(dataset) if dataset else None,
        },
        "artifacts": {
            "base_model_config_sha256": (
                optional_file_sha256(model_path / "config.json") if model_path else None
            ),
            "lora_adapter_config_sha256": (
                optional_file_sha256(adapter_path / "adapter_config.json") if adapter_path else None
            ),
            "lora_weights_file": str(adapter_weights.resolve()) if adapter_weights else None,
            "lora_weights_sha256": file_sha256(adapter_weights) if adapter_weights else None,
        },
    }


def write_run_manifest(path: Path, experiment_config: dict) -> None:
    payload = {
        "schema_version": "local-repair-run-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_config": experiment_config,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment_config") != experiment_config:
            raise ValueError(
                f"Existing run manifest has different settings: {path}. "
                "Choose a new --run_tag/output directory."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args) -> None:
    source = Path(args.input)
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Repair output must differ from the source trajectory file")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and truncate_incomplete_jsonl_tail(output):
        print(f"Removed an incomplete trailing JSONL record from {output}", flush=True)
    if args.continuation_model_type != "teacher" and not args.continuation_model_id:
        raise ValueError("--continuation_model_id is required unless continuation_model_type=teacher")
    experiment_config = build_experiment_config(args, source)
    write_run_manifest(Path(args.run_manifest), experiment_config)

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
        experiment_config=experiment_config,
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
                    "schema_version": "local-repair-v2",
                    "repair_id": current_id,
                    "question": entry.get("question"),
                    "accepted": False,
                    "retryable_error": True,
                    "rejection_reason": f"pipeline_error: {type(error).__name__}: {error}",
                    "experiment_config": experiment_config,
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
    parser.add_argument("--teacher_max_tokens", type=int, default=2048)
    parser.add_argument(
        "--continuation_model_type",
        choices=["teacher", "transformers", "openai"],
        default="transformers",
        help="Use 'transformers' plus the trained S0 model for the research setting.",
    )
    parser.add_argument("--continuation_model_id")
    parser.add_argument("--continuation_lora_path")
    parser.add_argument("--continuation_max_tokens", type=int, default=2048)
    parser.add_argument("--max_candidates", type=int, default=5)
    parser.add_argument("--max_continuation_steps", type=int, default=4)
    parser.add_argument("--max_format_retries", type=int, default=1)
    parser.add_argument("--execution_timeout_seconds", type=int, default=30)
    parser.add_argument("--max_entries", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_tag", default="repair-v2")
    parser.add_argument("--run_manifest", required=True)
    parser.add_argument("--dataset_path")
    parser.add_argument("--student_evaluation_max_tokens", type=int, default=2048)
    parser.add_argument("--student_evaluation_max_steps", type=int, default=5)
    parser.add_argument("--student_evaluation_max_samples", type=int, default=-1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
