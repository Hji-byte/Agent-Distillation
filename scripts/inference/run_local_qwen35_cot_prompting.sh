#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

model_path="${1:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
data_path="${2:-data_processor/math_dataset/test/math_500_20250414.json}"
max_samples="${3:-500}"
max_tokens="${4:-4096}"
log_folder="${5:-logs/qa_results/transformers/qwen3.5-0.8B_cot_prompting}"

if [[ -z "$model_path" ]]; then
    echo "Model path is required. Pass it as argument 1 or set AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi
if [[ ! -f "$data_path" ]]; then
    echo "Dataset was not found at $data_path" >&2
    exit 2
fi

"$python_bin" -u -m exps_research.unified_framework.run_reasoning \
    --data_path "$data_path" \
    --model_type transformers \
    --model_id "$model_path" \
    --log_folder "$log_folder" \
    --task_type math \
    --max_tokens "$max_tokens" \
    --max_samples "$max_samples" \
    --n 1 \
    --temperature 0.0 \
    --seed 42 \
    --one_attempt_per_question \
    --suffix cot_prompting_base
