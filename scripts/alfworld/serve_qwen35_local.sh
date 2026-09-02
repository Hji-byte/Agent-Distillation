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
lora_path="${2:-}"

if [[ -z "$model_path" ]]; then
    echo "Model path is required as argument 1 or AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi
if [[ ! -d "$model_path" ]]; then
    echo "Model directory not found: $model_path" >&2
    exit 2
fi
if [[ -n "$lora_path" && ! -d "$lora_path" ]]; then
    echo "LoRA directory not found: $lora_path" >&2
    exit 2
fi

args=(
    -u -m exps_research.alfworld_eto.openai_compat_server
    --model-path "$model_path"
    --served-model-name qwen3.5-0.8b-local
    --host 127.0.0.1
    --port 8000
    --max-request-tokens 512
    --device-map cuda
    --seed 42
)
if [[ -n "$lora_path" ]]; then
    args+=(--lora-path "$lora_path")
fi

exec "$python_bin" "${args[@]}"
