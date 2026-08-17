#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

adapter_path="${1:-}"
model_path="${2:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
data_path="${3:-data_processor/math_dataset/test/math_500_20250414.json}"
max_samples="${4:-500}"
max_tokens="${5:-1280}"
retry_max_tokens="${6:-2048}"

if [[ -z "$adapter_path" || ! -f "$adapter_path/adapter_config.json" ]]; then
    echo "Argument 1 must be a LoRA adapter directory containing adapter_config.json." >&2
    exit 2
fi
if [[ -z "$model_path" ]]; then
    echo "Model path is required. Pass it as argument 2 or set AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi
if (( retry_max_tokens <= max_tokens )); then
    echo "retry_max_tokens must be greater than max_tokens." >&2
    exit 2
fi
if [[ ! -f "$data_path" ]]; then
    echo "Math500 dataset was not found at $data_path" >&2
    exit 2
fi

"$python_bin" -u -m exps_research.unified_framework.run_experiment \
    --experiment_type agent \
    --data_path "$data_path" \
    --model_type transformers \
    --model_id "$model_path" \
    --fine_tuned \
    --lora_folder "$adapter_path" \
    --max_tokens "$max_tokens" \
    --retry_max_tokens "$retry_max_tokens" \
    --max_steps 5 \
    --max_samples "$max_samples" \
    --task_type math \
    --n 1 \
    --temperature 0.0 \
    --seed 42 \
    --suffix v126_sft_qlora
