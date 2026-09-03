#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

split="${1:-test}"
mode="${2:-full}"
model_tag="${3:-base}"
eto_root="${ETO_ROOT:-$project_root/_local/upstream/ETO}"

if [[ ! "$model_tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Model tag may contain only letters, numbers, '.', '_' and '-'." >&2
    exit 2
fi

exp_name="_qwen35_local_${model_tag}_react2_system_user_${split}"

args=(
    -u -m exps_research.alfworld_eto.run_upstream
    --eto-root "$eto_root"
    --split "$split"
    --prompt-profile react-type-2shot-system-user
    --trajectory-format prompt-normalized
    --exp-name "$exp_name"
    --verbose
)
if [[ "$mode" == "smoke" ]]; then
    args+=(--debug --exp-name "${exp_name}_smoke")
elif [[ "$mode" != "full" ]]; then
    echo "Mode must be 'full' or 'smoke'." >&2
    exit 2
fi

exec "$python_bin" "${args[@]}"
