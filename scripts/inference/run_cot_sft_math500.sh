#!/usr/bin/env bash
# Evaluate the paired CoT SFT adapters, not the old agent policy.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/inference/run_cot_sft_math500.sh MODEL [both|normal|shortest] [--dry-run]"
  echo "Overrides: PYTHON, COT_SFT_ROOT, MATH500_DATA, BATCH_SIZE, MAX_TOKENS, CUDA_VISIBLE_DEVICES"
  exit 0
fi
model="${1:-${AGENT_DISTILLATION_MODEL_PATH:-}}"
variant="${2:-both}"
dry_run="${3:-}"
[[ -n "$model" ]] || { echo 'Provide the base model path.' >&2; exit 2; }
case "$variant" in
  both) variants=(normal shortest) ;;
  normal|shortest) variants=("$variant") ;;
  *) echo 'Choose both, normal, or shortest.' >&2; exit 2 ;;
esac
[[ "$dry_run" == "" || "$dry_run" == "--dry-run" ]] || { echo 'Unknown argument.' >&2; exit 2; }
[[ $# -le 3 ]] || { echo 'Too many arguments.' >&2; exit 2; }
python_bin="${PYTHON:-.venv/bin/python}"
root="${COT_SFT_ROOT:-training_outputs/Qwen3.5-0.8B}"
dataset="${MATH500_DATA:-data_processor/math_dataset/test/math_500_20250414.json}"
batch_size="${BATCH_SIZE:-4}"
tokens="${MAX_TOKENS:-4096}"
for value in "$batch_size" "$tokens"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Sample/token limits must be positive integers.' >&2; exit 2; }
done
[[ -f "$dataset" ]] || { echo "Dataset missing: $dataset" >&2; exit 2; }
# Validate both adapters before starting an expensive run.
if [[ "$dry_run" != "--dry-run" ]]; then
  command -v "$python_bin" >/dev/null || { echo "Python not found: $python_bin" >&2; exit 2; }
  [[ -f "$model/config.json" ]] || { echo "Base model config missing: $model/config.json" >&2; exit 2; }
  for item in "${variants[@]}"; do
    adapter="$root/cot_${item}_max4096_qlora_seed42/final"
    [[ -f "$adapter/adapter_config.json" ]] || { echo "Adapter missing: $adapter" >&2; exit 2; }
  done
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for item in "${variants[@]}"; do
  adapter="$root/cot_${item}_max4096_qlora_seed42/final"
  command_args=("$python_bin" -u -m scripts.inference.batched_transformers_reasoning
    --dataset "$dataset" --model "$model" --adapter "$adapter"
    --output "$adapter/qa_results/math500_cot_${item}_sft_max${tokens}_batch${batch_size}.jsonl"
    --max-new-tokens "$tokens" --batch-size "$batch_size")
  echo "CoT evaluation: $item; batch_size=$batch_size; max_new_tokens=$tokens; thinking=False"
  echo "Results: $adapter/qa_results/"
  if [[ "$dry_run" == "--dry-run" ]]; then
    printf '%q ' "${command_args[@]}"
    printf '\n'
  else
    "${command_args[@]}"
  fi
done
# This batch runner saves generations and token counts, not accuracy scores.
# Use the corrected scoring policy for final comparisons.
