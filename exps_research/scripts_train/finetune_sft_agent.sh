#!/bin/bash

set -e
set -x

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

model="${1:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
datapath="${2:-data_processor/processed/sft/qwen35_27b_math_medium_hard_1646_v126.jsonl}"
postfix="${3:-qlora}"
epoch="${4:-2}"
resume_from_checkpoint="${5:-}"

if [[ -z "$model" ]]; then
    echo "Model path is required. Pass it as argument 1 or set AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi
if [[ ! -f "$datapath" ]]; then
    echo "Training data was not found at $datapath" >&2
    exit 2
fi

resume_args=()
if [[ -n "$resume_from_checkpoint" ]]; then
    resume_args+=(--resume_from_checkpoint "$resume_from_checkpoint")
fi

"$python_bin" -u exps_research/finetune_sft.py \
    --model_name "$model" \
    --num_epochs "$epoch" \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 25 \
    --save_total_limit 2 \
    --lr 2e-4 \
    --train_filepath "$datapath" \
    --postfix "$postfix" \
    --solution_type agent \
    --use_qlora \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --optim adamw_torch_fused \
    --max_length 4096 \
    "${resume_args[@]}"
