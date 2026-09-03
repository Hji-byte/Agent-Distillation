#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

split="${1:-train}"
mode="${2:-smoke}"
api_base="${ALFWORLD_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
model_name="${ALFWORLD_TEACHER_MODEL:-qwen3.5-27b}"
eto_root="${ETO_ROOT:-$project_root/_local/upstream/ETO}"
task_manifest="${ALFWORLD_TASK_MANIFEST:-$project_root/data_processor/alfworld_dataset/train/alfworld_train_teacher_half_seed42.json}"
exp_name="_teacher_api_react2_system_user"

args=(
    -u -m exps_research.alfworld_eto.run_upstream
    --eto-root "$eto_root"
    --split "$split"
    --api-base "$api_base"
    --api-key-env ALFWORLD_API_KEY
    --model-name "$model_name"
    --max-tokens 1024
    --temperature 0.0
    --prompt-profile react-type-2shot-system-user
    --trajectory-format prompt-normalized
    --verbose
    --exp-name "$exp_name"
)
if [[ "$split" == "train" ]]; then
    args+=(--task-manifest "$task_manifest")
fi
if [[ "$mode" == "one" ]]; then
    args+=(--max-tasks 1 --exp-name "${exp_name}_one")
elif [[ "$mode" == "probe" ]]; then
    args+=(--max-tasks 1 --exp-name "${exp_name}_probe")
elif [[ "$mode" == "five" ]]; then
    # Reusable pilot: writes to the formal output directory so a later full
    # run skips these same manifest task IDs instead of generating them again.
    args+=(--max-tasks 5 --exp-name "$exp_name")
elif [[ "$mode" == "fifty" ]]; then
    # The 50-task pilot is the first shard of the formal run.  Keeping the
    # final experiment name lets ETO reuse these task IDs when `full` resumes.
    args+=(--max-tasks 50 --exp-name "$exp_name")
elif [[ "$mode" == "five-hundred" ]]; then
    # Reusable 500-task shard of the formal run. Existing task IDs in the
    # formal output directory are skipped, including an earlier 5-task pilot.
    args+=(--max-tasks 500 --exp-name "$exp_name")
elif [[ "$mode" == "smoke" ]]; then
    args+=(--debug --exp-name "${exp_name}_smoke")
elif [[ "$mode" != "full" ]]; then
    echo "Mode must be 'one', 'probe', 'five', 'fifty', 'five-hundred', 'smoke', or 'full'." >&2
    exit 2
fi

exec "$python_bin" "${args[@]}"
