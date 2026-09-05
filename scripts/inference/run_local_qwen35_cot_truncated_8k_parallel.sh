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
run_root="${4:-logs/qa_results/vllm/qwen3.5-0.8B_cot_truncated_8k}"

source_max_tokens=4096
rerun_max_tokens=8192
parallel_workers="${AGENT_DISTILLATION_PARALLEL_WORKERS:-8}"
max_model_len="${AGENT_DISTILLATION_MAX_MODEL_LEN:-12288}"
gpu_memory_utilization="${AGENT_DISTILLATION_GPU_MEMORY_UTILIZATION:-0.90}"
server_port="${AGENT_DISTILLATION_VLLM_PORT:-8000}"
export VLLM_API_BASE="http://127.0.0.1:$server_port/v1"
server_log="$run_root/vllm_server.log"
rerun_dataset="$run_root/math500_truncated_at_4096.json"
rerun_log_root="$run_root/rerun"
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
if ! [[ "$parallel_workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENT_DISTILLATION_PARALLEL_WORKERS must be a positive integer." >&2
    exit 2
fi
if ! "$python_bin" -c 'import vllm' >/dev/null 2>&1; then
    echo "vLLM is not installed in the selected Python environment: $python_bin" >&2
    echo "Install a vLLM release compatible with Qwen3.5, then rerun this script." >&2
    exit 2
fi

mkdir -p "$run_root" "$rerun_log_root"

"$python_bin" scripts/analysis/truncated_cot_rerun.py prepare \
    --dataset "$dataset_path" \
    --baseline "$baseline_4k" \
    --output "$rerun_dataset" \
    --source-max-tokens "$source_max_tokens" \
    --rerun-max-tokens "$rerun_max_tokens"

rerun_count="$($python_bin -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["examples"]))' "$rerun_dataset")"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting one-GPU vLLM server for $parallel_workers concurrent requests..."
"$python_bin" -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name "$model_path" \
    --host 127.0.0.1 \
    --port "$server_port" \
    --dtype auto \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$parallel_workers" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --enable-prefix-caching \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >"$server_log" 2>&1 &
server_pid=$!

server_ready=0
for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:$server_port/v1/models" >/dev/null; then
        server_ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "vLLM exited before becoming ready. Last server log lines:" >&2
        tail -n 80 "$server_log" >&2 || true
        exit 2
    fi
    sleep 2
done
if [[ "$server_ready" -ne 1 ]]; then
    echo "vLLM did not become ready within 360 seconds. Last server log lines:" >&2
    tail -n 80 "$server_log" >&2 || true
    exit 2
fi

"$python_bin" -u -m exps_research.unified_framework.run_reasoning \
    --data_path "$rerun_dataset" \
    --model_type vllm \
    --model_id "$model_path" \
    --log_folder "$rerun_log_root" \
    --task_type math \
    --max_tokens "$rerun_max_tokens" \
    --max_samples "$rerun_count" \
    --n 1 \
    --temperature 0.0 \
    --seed 42 \
    --one_attempt_per_question \
    --multithreading \
    --parallel_workers "$parallel_workers" \
    --use_single_endpoint \
    --suffix cot_truncated_only_8k

mapfile -t rerun_candidates < <(
    find "$rerun_log_root" -type f \
        -name '*max_tokens=8192_cot_truncated_only_8k.jsonl' \
        ! -path '*/evaluations/*'
)
if [[ "${#rerun_candidates[@]}" -ne 1 ]]; then
    echo "Expected exactly one 8K rerun result, found ${#rerun_candidates[@]}." >&2
    printf '%s\n' "${rerun_candidates[@]}" >&2
    exit 2
fi
rerun_raw="${rerun_candidates[0]}"

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

echo "Parallel 8K truncated-only CoT experiment complete."
echo "Workers: $parallel_workers"
echo "Merged raw results: $merged_raw"
echo "Merge summary: $merge_summary"
echo "Scored results: $run_root/evaluations"
