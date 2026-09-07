#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${1:?Usage: bash scripts/training/run_cot_sft_4096.sh MODEL normal|shortest [--preflight|--smoke|--resume]}"
variant="${2:?Choose normal or shortest}"
shift 2
case "$variant" in normal|shortest) ;; *) echo 'Choose normal or shortest' >&2; exit 2;; esac
python_bin="${PYTHON:-.venv/bin/python}"
suffix=""
for arg in "$@"; do
    if [[ "$arg" == "--smoke" ]]; then suffix="_smoke"; fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Reduce allocator fragmentation; this does not guarantee sufficient memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$python_bin" -u -m scripts.training.train_cot_pair \
  --model "$model" --variant "$variant" \
  --output "training_outputs/Qwen3.5-0.8B/cot_${variant}_max4096_qlora_seed42${suffix}" \
  "$@" --max-length 4096
