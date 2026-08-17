"""Continue an S0 LoRA using only counterfactually verified repair targets.

This is a separate training entry point. It never changes or overwrites the
baseline trainer/checkpoint unless the caller explicitly chooses such a path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from exps_research.repair.sft import tokenize_last_assistant_only


def load_examples(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("supervision") != "last_assistant_only":
                raise ValueError("Repair trainer accepts only last_assistant_only examples")
            rows.append(row)
    if not rows:
        raise ValueError("Repair SFT dataset is empty")
    return Dataset.from_list(rows)


def qwen35_target_modules() -> list[str]:
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def main(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    if args.student_lora and output_dir == Path(args.student_lora).resolve():
        raise ValueError("output_dir must differ from the baseline S0 LoRA directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    quantization = None
    model_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if args.use_qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA repair training requires CUDA")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs.update(
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
        )

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    if args.student_lora:
        model = PeftModel.from_pretrained(model, args.student_lora, is_trainable=True)
    else:
        target_modules = qwen35_target_modules() if config.model_type == "qwen3_5" else "all-linear"
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_examples(Path(args.train_filepath))
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda row: tokenize_last_assistant_only(
            tokenizer,
            row["messages"],
            max_length=args.max_length,
        ),
        remove_columns=original_columns,
        desc="Tokenizing verified local repairs",
    )
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
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Base student model, e.g. Qwen3.5-0.8B")
    parser.add_argument("--student_lora", help="Baseline S0 LoRA to continue training")
    parser.add_argument("--train_filepath", required=True, help="local-repair-sft-v1 JSONL")
    parser.add_argument("--output_dir", required=True, help="New S1 adapter directory")
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
