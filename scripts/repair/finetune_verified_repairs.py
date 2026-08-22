"""Train the two registered verifier-grounded repair experiments.

``mixed_retrain`` starts from the base student and jointly trains ordinary
teacher trajectories plus verified repair targets. ``incremental_repair``
continues the S0 adapter using verified repair targets only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from exps_research.repair.sft import tokenize_supervised_messages


EXPERIMENT_MODES = {"mixed_retrain", "incremental_repair"}


def apply_mode_defaults(args) -> None:
    """Fill only mode-dependent defaults, preserving explicit CLI overrides."""
    if args.num_epochs is None:
        args.num_epochs = 2.0 if args.experiment_mode == "mixed_retrain" else 1.0
    if args.lr is None:
        args.lr = 2e-4 if args.experiment_mode == "mixed_retrain" else 5e-5
    if args.save_steps is None:
        args.save_steps = 25 if args.experiment_mode == "mixed_retrain" else 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_records(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    artifacts = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    return [
        {
            "file": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifacts
    ]


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_record() -> dict[str, Any]:
    status_short = _git_value("status", "--short") or ""
    tracked_diff = _git_value("diff", "--no-ext-diff", "HEAD") or ""
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "working_tree_dirty": bool(status_short),
        "status_short": status_short.splitlines(),
        "tracked_diff_sha256": _text_sha256(tracked_diff),
        "tracked_diff": tracked_diff or None,
    }


def _code_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "commit": record.get("commit"),
        "tracked_diff_sha256": record.get("tracked_diff_sha256"),
    }


def _model_record(value: str) -> dict[str, Any]:
    path = Path(value)
    if not path.exists():
        return {"model_id": value}
    resolved = path.resolve()
    record: dict[str, Any] = {"path": str(resolved)}
    record["configuration_artifacts"] = _artifact_records(
        resolved,
        ("config.json", "generation_config.json", "*.index.json"),
    )
    record["weight_artifacts"] = _artifact_records(
        resolved,
        ("*.safetensors", "pytorch_model*.bin"),
    )
    return record


def _tokenizer_record(tokenizer, model_value: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "chat_template_sha256": _text_sha256(tokenizer.chat_template or ""),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    root = Path(model_value)
    if root.exists():
        record["artifacts"] = _artifact_records(
            root.resolve(),
            (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ),
        )
    return record


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "datasets", "bitsandbytes", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _effective_lora_record(args) -> dict[str, Any]:
    if args.experiment_mode == "mixed_retrain":
        return {
            "source": "new_adapter",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": qwen35_target_modules(),
        }
    config_path = Path(args.student_lora).resolve() / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"S0 adapter config is missing: {config_path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "source": "inherited_s0_adapter",
        "r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_dropout": adapter_config.get("lora_dropout"),
        "target_modules": adapter_config.get("target_modules"),
    }


def _adapter_record(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    resolved = Path(value).resolve()
    record: dict[str, Any] = {"path": str(resolved)}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        artifact = resolved / name
        if artifact.is_file():
            record[f"{name}_sha256"] = _sha256(artifact)
    return record


def _read_jsonl(path: Path, *, source: str, supervision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Invalid SFT row at {path}:{line_number}")
            declared = row.get("supervision")
            if source == "repair" and declared != "last_assistant_only":
                raise ValueError(
                    f"Repair row at {path}:{line_number} must declare last_assistant_only supervision"
                )
            if source == "repair":
                assistant_count = sum(
                    message.get("role") == "assistant" for message in row["messages"]
                )
                target_index = row.get("target_assistant_turn_index")
                if target_index != assistant_count - 1:
                    raise ValueError(
                        f"Repair row at {path}:{line_number} declares target assistant "
                        f"index {target_index!r}, but the final assistant index is {assistant_count - 1}"
                    )
            if source == "baseline" and declared not in {None, "all_assistant_turns"}:
                raise ValueError(
                    f"Baseline row at {path}:{line_number} has incompatible supervision={declared!r}"
                )
            rows.append(
                {
                    "messages": row["messages"],
                    "source": source,
                    "supervision": supervision,
                    "source_path": str(path.resolve()),
                    "source_line": line_number,
                }
            )
    if not rows:
        raise ValueError(f"SFT dataset is empty: {path}")
    return rows


def _task_text(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            content = message["content"].strip()
            if content:
                return content
    raise ValueError(
        f"SFT row has no non-empty user task: {row['source_path']}:{row['source_line']}"
    )


def load_experiment_rows(
    *,
    experiment_mode: str,
    baseline_filepath: Path | None,
    repair_filepaths: list[Path],
    student_lora: Path | None,
) -> list[dict[str, Any]]:
    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment_mode={experiment_mode!r}")
    if experiment_mode == "mixed_retrain":
        if baseline_filepath is None:
            raise ValueError("mixed_retrain requires --baseline_filepath")
        if student_lora is not None:
            raise ValueError("mixed_retrain starts from the base model and must not receive --student_lora")
        rows = _read_jsonl(
            baseline_filepath,
            source="baseline",
            supervision="all_assistant_turns",
        )
    else:
        if baseline_filepath is not None:
            raise ValueError("incremental_repair trains repair-only and must not receive --baseline_filepath")
        if student_lora is None:
            raise ValueError("incremental_repair requires the S0 --student_lora")
        rows = []

    if not repair_filepaths:
        raise ValueError("At least one --repair_filepath is required")
    for repair_filepath in repair_filepaths:
        rows.extend(
            _read_jsonl(
                repair_filepath,
                source="repair",
                supervision="last_assistant_only",
            )
        )
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = _task_text(row)
        if task in seen:
            previous = seen[task]
            raise ValueError(
                "Duplicate task across training inputs: "
                f"{previous['source_path']}:{previous['source_line']} and "
                f"{row['source_path']}:{row['source_line']}"
            )
        seen[task] = row
    return rows


def tokenize_experiment_rows(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    max_assistant_tokens: int,
) -> tuple[Dataset, dict[str, Any]]:
    tokenized_rows: list[dict[str, list[int]]] = []
    source_counts: dict[str, int] = {}
    source_supervised_tokens: dict[str, int] = {}
    lengths: list[int] = []
    for row in rows:
        try:
            encoded = tokenize_supervised_messages(
                tokenizer,
                row["messages"],
                supervision=row["supervision"],
                max_length=max_length,
                max_assistant_tokens=max_assistant_tokens,
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid {row['source']} training row at "
                f"{row['source_path']}:{row['source_line']}: {error}"
            ) from error
        source = row["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
        source_supervised_tokens[source] = source_supervised_tokens.get(source, 0) + int(
            encoded.pop("supervised_token_count")
        )
        lengths.append(int(encoded.pop("sequence_length")))
        tokenized_rows.append(encoded)
    stats = {
        "total_examples": len(tokenized_rows),
        "examples_by_source": source_counts,
        "supervised_tokens_by_source": source_supervised_tokens,
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
        "sequence_length_mean": sum(lengths) / len(lengths),
    }
    return Dataset.from_list(tokenized_rows), stats


def qwen35_target_modules() -> list[str]:
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def _resolve_resume(output_dir: Path, requested: str | None) -> str | None:
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    if requested is None:
        non_manifest = [path for path in existing if path.name != "repair_training_manifest.json"]
        if non_manifest:
            raise ValueError(
                f"output_dir is not empty: {output_dir}. Use a new directory or "
                "--resume_from_checkpoint latest."
            )
        return None
    if not existing:
        raise ValueError("Cannot resume because output_dir is empty")
    if requested.lower() == "latest":
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint is None:
            raise ValueError(f"No checkpoint-* directory found under {output_dir}")
        return checkpoint
    checkpoint = Path(requested).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    if output_dir not in checkpoint.parents:
        raise ValueError("Resume checkpoint must be inside output_dir")
    return str(checkpoint)


def _experiment_config(
    args,
    *,
    baseline_path: Path | None,
    repair_paths: list[Path],
    tokenizer,
    precision: str,
    optimizer: str,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "repairs": [
            {"path": str(path), "sha256": _sha256(path)} for path in repair_paths
        ],
    }
    if baseline_path is not None:
        inputs["baseline"] = {"path": str(baseline_path), "sha256": _sha256(baseline_path)}
    return {
        "schema_version": "repair-training-config-v2",
        "experiment_mode": args.experiment_mode,
        "base_model": _model_record(args.model_name),
        "tokenizer": _tokenizer_record(tokenizer, args.model_name),
        "student_lora": _adapter_record(args.student_lora),
        "inputs": inputs,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_assistant_tokens": args.max_assistant_tokens,
        "effective_lora": _effective_lora_record(args),
        "seed": args.seed,
        "use_qlora": args.use_qlora,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "precision": precision,
        "optimizer": optimizer,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None,
            "packages": _package_versions(),
        },
    }


def _write_or_validate_manifest(
    output_dir: Path,
    config: dict[str, Any],
    data_stats: dict[str, Any],
    *,
    resuming: bool,
) -> None:
    manifest_path = output_dir / "repair_training_manifest.json"
    current_code = _code_record()
    if resuming:
        if not manifest_path.is_file():
            raise ValueError(f"Resume manifest is missing: {manifest_path}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("experiment_config") != config:
            raise ValueError("Current training settings or input hashes differ from the existing run manifest")
        if _code_identity(previous.get("code") or {}) != _code_identity(current_code):
            raise ValueError(
                "Current Git commit or tracked code diff differs from the run being resumed"
            )
        return
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("experiment_config") != config:
            raise ValueError("Existing preflight manifest belongs to a different training configuration")
        return
    manifest = {
        "schema_version": "repair-training-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": config,
        "data_stats": data_stats,
        "code": current_code,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _finalize_manifest(output_dir: Path, trainer: Trainer, train_result) -> None:
    manifest_path = output_dir / "repair_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_result"] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "metrics": train_result.metrics,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args) -> None:
    repair_paths = [Path(path).resolve() for path in args.repair_filepath]
    baseline_path = Path(args.baseline_filepath).resolve() if args.baseline_filepath else None
    student_lora = Path(args.student_lora).resolve() if args.student_lora else None
    output_dir = Path(args.output_dir).resolve()
    if student_lora is not None and output_dir == student_lora:
        raise ValueError("output_dir must differ from the baseline S0 LoRA directory")
    resume_checkpoint = _resolve_resume(output_dir, args.resume_from_checkpoint)

    rows = load_experiment_rows(
        experiment_mode=args.experiment_mode,
        baseline_filepath=baseline_path,
        repair_filepaths=repair_paths,
        student_lora=student_lora,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, data_stats = tokenize_experiment_rows(
        tokenizer,
        rows,
        max_length=args.max_length,
        max_assistant_tokens=args.max_assistant_tokens,
    )

    if args.use_qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA repair training requires CUDA")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    optimizer = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    output_dir.mkdir(parents=True, exist_ok=True)
    config = _experiment_config(
        args,
        baseline_path=baseline_path,
        repair_paths=repair_paths,
        tokenizer=tokenizer,
        precision=precision,
        optimizer=optimizer,
    )
    _write_or_validate_manifest(
        output_dir,
        config,
        data_stats,
        resuming=resume_checkpoint is not None,
    )
    print(json.dumps({"experiment_config": config, "data_stats": data_stats}, ensure_ascii=False, indent=2))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Training precision: {compute_dtype}")

    model_kwargs: dict[str, Any] = {"dtype": compute_dtype, "trust_remote_code": True}
    if args.use_qlora:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map={"": torch.cuda.current_device()},
        )

    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    if args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if args.experiment_mode == "incremental_repair":
        model = PeftModel.from_pretrained(model, str(student_lora), is_trainable=True)
    else:
        target_modules = qwen35_target_modules() if model_config.model_type == "qwen3_5" else "all-linear"
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=optimizer,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _finalize_manifest(output_dir, trainer, train_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_mode", choices=sorted(EXPERIMENT_MODES), required=True)
    parser.add_argument("--model_name", required=True, help="Base student model, e.g. Qwen3.5-0.8B")
    parser.add_argument("--student_lora", help="S0 LoRA; required only for incremental_repair")
    parser.add_argument("--baseline_filepath", help="Ordinary 1646-trace SFT JSONL; mixed_retrain only")
    parser.add_argument(
        "--repair_filepath",
        action="append",
        help="Verified local-repair SFT JSONL; repeat to combine smoke and formal repairs",
    )
    parser.add_argument("--train_filepath", dest="legacy_repair_filepath", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", required=True, help="New experiment adapter directory")
    parser.add_argument("--num_epochs", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_assistant_tokens", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Positive values override epochs; intended for smoke tests",
    )
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", help="Use 'latest' or a checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parsed = parser.parse_args()
    if parsed.repair_filepath and parsed.legacy_repair_filepath:
        parser.error("Use only --repair_filepath (the old --train_filepath is an alias)")
    parsed.repair_filepath = parsed.repair_filepath or (
        [parsed.legacy_repair_filepath] if parsed.legacy_repair_filepath else None
    )
    if not parsed.repair_filepath:
        parser.error("--repair_filepath is required")
    apply_mode_defaults(parsed)
    main(parsed)
