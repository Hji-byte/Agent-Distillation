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
baseline_4k="${2:-experiment_results/math500/qwen3.5-0.8B_cot_prompting/math_500_20250414_test/Qwen3.5-0.8B_temp=0.0_seed=42_type=reasoning_max_tokens=4096_cot_prompting_base.jsonl}"
dataset_path="${3:-data_processor/math_dataset/test/math_500_20250414.json}"
run_root="${4:-logs/qa_results/transformers/qwen3.5-0.8B_cot_truncated_8k}"

source_max_tokens=4096
rerun_max_tokens=8192
batch_size="${AGENT_DISTILLATION_BATCH_SIZE:-4}"
rerun_dataset="$run_root/math500_truncated_at_4096.json"
rerun_log_root="$run_root/rerun"
rerun_raw="$run_root/math500_cot_truncated_only_8k_batched.jsonl"
merged_raw="$run_root/math500_cot_4k_plus_truncated_8k.jsonl"
merge_summary="$run_root/math500_cot_4k_plus_truncated_8k.merge_summary.json"

if [[ -z "$model_path" ]]; then
    echo "Model path is required. Pass it as argument 1 or set AGENT_DISTILLATION_MODEL_PATH." >&2
    exit 2
fi
for required_path in "$model_path/config.json" "$baseline_4k" "$dataset_path"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Required file was not found: $required_path" >&2
        exit 2
    fi
done
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENT_DISTILLATION_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi

mkdir -p "$run_root" "$rerun_log_root"

"$python_bin" scripts/analysis/truncated_cot_rerun.py prepare \
    --dataset "$dataset_path" \
    --baseline "$baseline_4k" \
    --output "$rerun_dataset" \
    --source-max-tokens "$source_max_tokens" \
    --rerun-max-tokens "$rerun_max_tokens"

resume_args=()
while IFS= read -r candidate; do
    if [[ "$candidate" != "$rerun_raw" ]]; then
        resume_args+=(--resume-from "$candidate")
    fi
done < <(
    find "$rerun_log_root" -type f \
        -name '*max_tokens=8192_cot_truncated_only_8k.jsonl' \
        ! -path '*/evaluations/*' \
        -print
)

echo "Starting batched Transformers generation with batch size $batch_size."
echo "Found ${#resume_args[@]} resume arguments from earlier sequential runs."
"$python_bin" -u scripts/inference/batched_transformers_reasoning.py \
    --dataset "$rerun_dataset" \
    --model "$model_path" \
    --output "$rerun_raw" \
    --batch-size "$batch_size" \
    --max-new-tokens "$rerun_max_tokens" \
    "${resume_args[@]}"

"$python_bin" scripts/analysis/truncated_cot_rerun.py merge \
    --dataset "$dataset_path" \
    --baseline "$baseline_4k" \
    --rerun "$rerun_raw" \
    --output "$merged_raw" \
    --summary "$merge_summary" \
    --source-max-tokens "$source_max_tokens" \
    --rerun-max-tokens "$rerun_max_tokens"

"$python_bin" -m exps_research.unified_framework.score_answers \
    --log_files "$merged_raw" \
    --task_type math \
    --single_thread \
    --attempt_selection first

echo "Batched 8K truncated-only CoT experiment complete."
echo "Batch size: $batch_size"
echo "Rerun results: $rerun_raw"
echo "Merged raw results: $merged_raw"
echo "Merge summary: $merge_summary"
echo "Scored results: $run_root/evaluations"
