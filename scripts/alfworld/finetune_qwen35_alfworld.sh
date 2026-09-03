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

model="${1:-}"
datapath="${2:-data_processor/processed/sft/alfworld_teacher_success_react2_system_user_sft_max6400.jsonl}"
postfix="${3:-alfworld_teacher_sft_max6400_qlora}"
epochs="${4:-2}"
max_length="${5:-6400}"

if [[ -z "$model" ]]; then
    echo "Model path is required as argument 1." >&2
    exit 2
fi
if [[ ! -f "$datapath" ]]; then
    echo "Training data was not found at $datapath" >&2
    exit 2
fi

"$python_bin" -u exps_research/finetune_sft.py \
    --model_name "$model" \
    --num_epochs "$epochs" \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 25 \
    --save_total_limit 2 \
    --lr 2e-4 \
    --train_filepath "$datapath" \
    --postfix "$postfix" \
    --solution_type alfworld \
    --use_qlora \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --optim adamw_torch_fused \
    --max_length "$max_length"
