#!/usr/bin/env bash

# Run S0 on a repair split, call Qwen3.5-27B only for local replacement
# actions, verify with local S0 continuations, and materialize accepted repairs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

adapter_path="${1:-${S0_ADAPTER_PATH:-}}"
model_path="${2:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
data_path="${3:-$project_root/data_processor/math_dataset/train/math_repair_smoke_50_seed42.json}"
max_samples="${4:-50}"
student_max_tokens="${STUDENT_MAX_TOKENS:-2048}"
teacher_model_id="${REPAIR_TEACHER_MODEL_ID:-qwen3.5-27b}"
teacher_max_tokens="${REPAIR_TEACHER_MAX_TOKENS:-1280}"
continuation_max_tokens="${REPAIR_CONTINUATION_MAX_TOKENS:-1024}"
max_candidates="${REPAIR_MAX_CANDIDATES:-5}"
max_continuation_steps="${REPAIR_MAX_CONTINUATION_STEPS:-4}"
repair_max_entries="${REPAIR_MAX_ENTRIES:--1}"

if [[ -z "$adapter_path" || ! -f "$adapter_path/adapter_config.json" ]]; then
    echo "Argument 1 must be the S0 LoRA directory containing adapter_config.json." >&2
    exit 2
fi
if [[ -z "$model_path" || ! -f "$model_path/config.json" ]]; then
    echo "Argument 2 must be the local Qwen3.5-0.8B directory containing config.json." >&2
    exit 2
fi
if [[ ! -f "$data_path" ]]; then
    echo "Repair dataset was not found: $data_path" >&2
    exit 2
fi
if ! [[ "$max_samples" =~ ^[1-9][0-9]*$ ]]; then
    echo "Argument 4 (max_samples) must be a positive integer." >&2
    exit 2
fi

# setup_model loads project_root/.env itself. This accepts either that file or
# exported variables without ever printing the secret key.
if ! "$python_bin" -c '
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env")
key = os.getenv("DASHSCOPE_API_KEY", "").strip()
base = os.getenv("DASHSCOPE_BASE_URL", "").strip()
if not key or key == "your_dashscope_api_key_here" or not base:
    raise SystemExit(1)
'; then
    echo "DASHSCOPE_API_KEY or DASHSCOPE_BASE_URL is missing." >&2
    echo "Configure them in $project_root/.env or export them in this shell." >&2
    exit 2
fi

dataset_file="$(basename "$data_path")"
dataset_stem="${dataset_file%.*}"
dataset_fold="$(basename "$(dirname "$data_path")")"
model_name="$(basename "${model_path%/}")"
teacher_name="${teacher_model_id//\//_}"
evaluation_dir="$adapter_path/qa_results/${dataset_stem}_${dataset_fold}"
evaluation_base="${model_name}_temp=0.0_n=1_seed=42_type=agent_steps=5_max_tokens=${student_max_tokens}_code_only_v126_sft_qlora"
scored_file="$(
    "$python_bin" -c '
import sys
from exps_research.unified_framework.path_utils import bounded_artifact_path
print(bounded_artifact_path(sys.argv[1], sys.argv[2], "_scored.jsonl"))
' "$evaluation_dir/evaluations" "$evaluation_base"
)"

repair_dir="$adapter_path/repair_results/${dataset_stem}"
attempts_file="$repair_dir/${teacher_name}_local_repair_attempts.jsonl"
verified_sft_file="$repair_dir/${teacher_name}_verified_repair_sft.jsonl"
mkdir -p "$repair_dir"

echo "[1/3] Running or resuming local S0 evaluation on: $data_path"
bash "$project_root/scripts/inference/run_local_qwen35_finetuned.sh" \
    "$adapter_path" \
    "$model_path" \
    "$data_path" \
    "$max_samples" \
    "$student_max_tokens"

if [[ ! -f "$scored_file" ]]; then
    echo "Expected scored S0 result was not found: $scored_file" >&2
    exit 2
fi

echo "[2/3] Repairing scored S0 failures with on-demand $teacher_model_id API calls"
"$python_bin" -u "$project_root/scripts/repair/generate_local_repairs.py" \
    --input "$scored_file" \
    --output "$attempts_file" \
    --teacher_model_id "$teacher_model_id" \
    --teacher_max_tokens "$teacher_max_tokens" \
    --continuation_model_type transformers \
    --continuation_model_id "$model_path" \
    --continuation_lora_path "$adapter_path" \
    --continuation_max_tokens "$continuation_max_tokens" \
    --max_candidates "$max_candidates" \
    --max_continuation_steps "$max_continuation_steps" \
    --max_entries "$repair_max_entries" \
    --max_format_retries 1 \
    --execution_timeout_seconds 30 \
    --seed 42 \
    --resume

echo "[3/3] Materializing only counterfactually verified repair targets"
"$python_bin" "$project_root/scripts/repair/materialize_repair_sft.py" \
    --input "$attempts_file" \
    --output "$verified_sft_file"

echo "Repair pipeline complete."
echo "S0 scored trajectories: $scored_file"
echo "All repair attempts:    $attempts_file"
echo "Verified repair SFT:    $verified_sft_file"
