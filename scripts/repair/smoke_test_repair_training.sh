#!/usr/bin/env bash

# Exercise both formal repair-training modes with real 4-bit QLoRA for a few
# optimizer steps. Outputs are isolated from S0 and formal experiment adapters.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

model_path="${1:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
s0_adapter_path="${2:-${S0_ADAPTER_PATH:-}}"
baseline_path="${REPAIR_BASELINE_SFT_PATH:-$project_root/data_processor/processed/sft/qwen35_27b_math_medium_hard_1646_v126.jsonl}"
repair_path="${REPAIR_TRAINABLE_SFT_PATH:-$project_root/experiment_results/repair/qwen3.5-0.8b_smoke50_train500_combined_v1/verified_repair_sft_trainable_4096.jsonl}"
smoke_steps="${REPAIR_TRAINING_SMOKE_STEPS:-2}"
run_tag="${REPAIR_TRAINING_SMOKE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="$project_root/training_outputs/Qwen3.5-0.8B/repair_training_smoke/$run_tag"

if [[ -z "$model_path" || ! -f "$model_path/config.json" ]]; then
    echo "Argument 1 or AGENT_DISTILLATION_MODEL_PATH must identify the base model." >&2
    exit 2
fi
if [[ -z "$s0_adapter_path" || ! -f "$s0_adapter_path/adapter_config.json" ]]; then
    echo "Argument 2 or S0_ADAPTER_PATH must identify the S0 LoRA adapter." >&2
    exit 2
fi
for input_path in "$baseline_path" "$repair_path"; do
    if [[ ! -f "$input_path" ]]; then
        echo "Required training input was not found: $input_path" >&2
        exit 2
    fi
done
if ! [[ "$smoke_steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPAIR_TRAINING_SMOKE_STEPS must be a positive integer." >&2
    exit 2
fi
if [[ -e "$output_root" ]]; then
    echo "Smoke output already exists; choose a new REPAIR_TRAINING_SMOKE_TAG: $output_root" >&2
    exit 2
fi

echo "[1/2] QLoRA smoke: Base -> ordinary trajectories + verified repairs"
"$python_bin" -u scripts/repair/finetune_verified_repairs.py \
    --experiment_mode mixed_retrain \
    --model_name "$model_path" \
    --baseline_filepath "$baseline_path" \
    --repair_filepath "$repair_path" \
    --output_dir "$output_root/mixed_retrain" \
    --max_steps "$smoke_steps" \
    --save_steps 1

echo "[2/2] QLoRA smoke: S0 -> verified repairs"
"$python_bin" -u scripts/repair/finetune_verified_repairs.py \
    --experiment_mode incremental_repair \
    --model_name "$model_path" \
    --student_lora "$s0_adapter_path" \
    --repair_filepath "$repair_path" \
    --output_dir "$output_root/incremental_repair" \
    --max_steps "$smoke_steps" \
    --save_steps 1

echo "Both QLoRA repair-training smoke tests completed."
echo "Smoke outputs: $output_root"
