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
if [[ -z "$model_path" ]]; then
    echo "Model path is required. Pass it as argument 1 or set AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi

dataset="$project_root/data_processor/math_dataset/test/math_500_20250414.json"
baseline_result_dir="$project_root/experiment_results/math500/qwen3.5-0.8B_baseline"
baseline_scored="$(
    find "$baseline_result_dir" -maxdepth 1 -type f \
        -name '*max_tokens=2048_code_only_v126_baseline_scored.jsonl' \
        -print -quit
)"
if [[ -z "$baseline_scored" ]]; then
    echo "The formal 2048-token baseline scored result was not found in $baseline_result_dir" >&2
    exit 2
fi

repair_dataset="$project_root/logs/repair_inputs/math_500_framework_errors_rerun.json"
repair_log_folder="$project_root/logs/qa_results/transformers/qwen3.5-0.8B_v126_baseline_framework_rerun"

"$python_bin" "$project_root/scripts/inference/prepare_framework_error_rerun.py" \
    --dataset "$dataset" \
    --results "$baseline_scored" \
    --output "$repair_dataset"

bash "$project_root/scripts/inference/run_local_qwen35_baseline.sh" \
    "$model_path" \
    "$repair_dataset" \
    7 \
    2048 \
    "$repair_log_folder"

echo "Framework-error rerun finished. Results are under:"
echo "$repair_log_folder"
