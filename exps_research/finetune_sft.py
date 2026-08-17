import os
import sys
sys.path.append(".")
import json
import torch
import random
import re
from pathlib import Path
from datetime import datetime

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AutoPeftModelForCausalLM
)

import argparse
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    DataCollatorForCompletionOnlyLM
)

from exps_research.train_utils.preprocess import (
    preprocess_sft_dataset,
)
from exps_research.train_utils.utils import DataCollatorForCompletionOnlyLMMultiTurn

MODEL_IDENTIFIERS = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama-1B-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3B-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-8B-instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-0.5B-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-1.5B-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-3B-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7B-instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "qwen-coder-1.5B-instruct",
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8B",
    "microsoft/Phi-3-mini-128k-instruct": "phi-3-mini-instruct",
    "microsoft/Phi-4-mini-instruct": "phi-4-mini-instruct",
}

def setup_savedir(args):
    # Step 3-1: Setup save dir
    if "training_outputs" in args.model_name:
        # Extract model_identifier from the path
        path_parts = args.model_name.split('/')

        # Find the part that might be a model identifier
        for part in path_parts:
            # Check if this part is a value in MODEL_IDENTIFIERS
            for model_name, identifier in MODEL_IDENTIFIERS.items():
                if part == identifier:
                    model_identifier = part
                    break

            # If we found a match, break out of the outer loop
            if 'model_identifier' in locals():
                break

        # If no match was found in the path, use a default
        if 'model_identifier' not in locals():
            # Try to infer from the directory structure
            if len(path_parts) >= 3 and path_parts[-3] == "training_outputs":
                model_identifier = path_parts[-2]
            else:
                model_identifier = "qwen-7B-instruct"  # Default fallback
    else:
        model_identifier = MODEL_IDENTIFIERS.get(args.model_name)
        if model_identifier is None:
            model_identifier = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(args.model_name).name)

    print(f"Model: {args.model_name}")

    if args.exp_id:
        exp_id = f"{args.solution_type}_{args.exp_id}"
    else:
        exp_id = f"{args.solution_type}_baseline"
        if args.num_epochs > 1:
            exp_id += f"_{args.num_epochs}epochs"
        if args.full_finetuning:
            exp_id += "_full"
        if len(args.postfix) > 0:
            if args.postfix.startswith("_"):
                exp_id += args.postfix
            else:
                exp_id += "_" + args.postfix

    # Keep smoke-test checkpoints separate from the full training run. Without
    # these suffixes, a short test can silently share an output directory with
    # the two-epoch experiment.
    if args.dataset_size > 0:
        exp_id += f"_start{args.dataset_start_index}_n{args.dataset_size}"
    if args.max_steps > 0:
        exp_id += f"_steps{args.max_steps}"

    output_dir = f"./training_outputs/{model_identifier}/{exp_id}"
    print("Output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata = vars(args)
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return output_dir
def main(args):
    # Set Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    is_qwen35 = model_config.model_type == "qwen3_5"
    is_qwen_model = model_config.model_type.startswith("qwen")

    if args.use_qlora and args.full_finetuning:
        raise ValueError("QLoRA can only train adapters; it cannot be combined with --full_finetuning.")
    if args.use_qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires an available CUDA GPU.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    )
    print(f"Training precision: {compute_dtype}")

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Math-agent SFT is text-only. Loading Qwen3.5 through its causal-LM class
    # avoids placing the unused vision encoder on the GPU and keeps the PEFT
    # module paths consistent with local student evaluation.
    model = None
    if args.use_qlora or args.peft_name or is_qwen35:
        model_kwargs = {
            "dtype": compute_dtype,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs.update(
                quantization_config=quantization_config,
                device_map={"": torch.cuda.current_device()},
            )
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

        if args.peft_name:
            model = PeftModel.from_pretrained(
                model,
                args.peft_name,
                is_trainable=True
            )

    if not args.full_finetuning and not args.peft_name:
        if is_qwen35:
            # Qwen3.5 mixes full-attention and linear-attention blocks, so
            # include the projection names used by both language-layer types.
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        else:
            target_modules = "all-linear"
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        peft_config = None

    # KV caching is useful for autoregressive inference, but it wastes memory
    # and is incompatible with activation recomputation during training.
    if args.gradient_checkpointing and model is not None:
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

    print("peft config", peft_config)

    ########## Setup model done ###############

    # Step 2: Setup dataset
    if is_qwen_model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, pad_token='<|endoftext|>', padding_side='left', add_eos_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left', add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = None
    for _train_filepath in args.train_filepath:
        _train_dataset = preprocess_sft_dataset(args.solution_type, _train_filepath)
        if train_dataset:
            train_dataset = concatenate_datasets([train_dataset, _train_dataset])
        else:
            train_dataset = _train_dataset
    if args.cot_filepath:
        _train_dataset = preprocess_sft_dataset("cot", args.cot_filepath)
        train_dataset = concatenate_datasets([train_dataset, _train_dataset]) # Add this

    if args.valid_filepath is not None:
        eval_dataset = preprocess_sft_dataset(args.solution_type, args.valid_filepath)
    else:
        eval_dataset = None

    if args.dataset_start_index < 0:
        raise ValueError("--dataset_start_index must be non-negative.")
    if args.dataset_start_index and args.dataset_size <= 0:
        raise ValueError("--dataset_start_index requires a positive --dataset_size.")
    if args.dataset_size > 0:
        start = args.dataset_start_index
        stop = min(start + args.dataset_size, len(train_dataset))
        if start >= stop:
            raise ValueError(
                f"Dataset slice [{start}:{stop}] is empty for {len(train_dataset)} rows."
            )
        train_dataset = train_dataset.select(range(start, stop))

    data_module = {
        "train_dataset": train_dataset
    }
    if eval_dataset is not None:
        data_module["eval_dataset"] = eval_dataset

    print("# Train Dataset: ", len(data_module["train_dataset"]))
    if "eval_dataset" in data_module.keys():
        print("# Valid Dataset: ", len(data_module["eval_dataset"]))
        eval_strategy = "epoch"
        save_strategy = "epoch"
        load_best_model_at_end = True
    else:
        eval_strategy = "no"
        save_strategy = "steps"
        load_best_model_at_end = False

    output_dir = setup_savedir(args)
    ########## Setup dataset done ###############

    batch_size = args.batch_size
    # Step 3: Train
    train_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=use_bf16,
        fp16=use_fp16,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp is not None,
        fsdp_config=args.fsdp,
        # Strategy
        logging_steps=10,
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        output_dir=output_dir,
        load_best_model_at_end=load_best_model_at_end,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )

    if is_qwen_model:
        response_template = "<|im_start|>assistant"
        instruction_template = "<|im_start|>user"
    elif "llama" in args.model_name.lower():
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
    elif "phi" in args.model_name.lower():
        instruction_template = "<|user|>"
        response_template = "<|assistant|>"
    else:
        raise NotImplementedError(f"Unsupported model {args.model_name} for response template")

    if args.solution_type == "agent":
        collator = DataCollatorForCompletionOnlyLMMultiTurn(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer
        )

    trainer = SFTTrainer(
        args.model_name if not model else model,
        args=train_args,
        peft_config=peft_config,
        data_collator=collator,
        processing_class=tokenizer,
        **data_module
    )
    resume_from_checkpoint = args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "latest":
        resume_from_checkpoint = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if torch.cuda.is_available():
        gib = 1024 ** 3
        print(
            "CUDA peak memory: "
            f"allocated={torch.cuda.max_memory_allocated() / gib:.2f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved() / gib:.2f} GiB"
        )
    ########## Train done ###############

    # Step 4: Save best model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Local model directory or remote model identifier.",
    )
    parser.add_argument("--peft_name", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument("--save_steps", default=25, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="Checkpoint directory to resume from, or 'latest' for the newest checkpoint in output_dir.",
    )
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--use_qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--lora_alpha", default=128, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--optim", default="adamw_torch_fused", type=str)
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument(
        "--dataset_start_index",
        default=0,
        type=int,
        help="First row to use when --dataset_size selects a smoke-test slice.",
    )
    parser.add_argument("--solution_type", type=str, default="agent", choices=["cot", "reasoning", "agent"])

    parser.add_argument(
        "--train_filepath",
        type=str,
        default=[(
            "data_processor/processed/sft/"
            "qwen35_27b_math_medium_hard_1646_v126.jsonl"
        )],
        nargs='+'
    )
    parser.add_argument(
        "--cot_filepath",
        type=str,
        help="Additional CoT dataset in agent training"
    )
    parser.add_argument("--valid_filepath", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)

    # Deepspeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)

    args = parser.parse_args()

    main(args)
