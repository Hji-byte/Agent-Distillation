# Paired CoT SFT, full sequence length 6400

Two independent QLoRA runs start from the original Qwen3.5-0.8B checkpoint:
`normal` uses a randomly selected correct teacher answer; `shortest` uses the
shortest correct teacher answer. Both contain the same 1940 questions, after
paired length filtering. Data lives in
`data/datasets/math/cot_teacher_sft_no_think_max6400_v4`.

## Cloud commands

Use the existing CUDA training environment with PyTorch, Transformers supporting
Qwen3.5, PEFT, bitsandbytes, datasets, and accelerate. vLLM is not required.
Do not blindly upgrade a working training environment.

```bash
cd /mnt/workspace/Agent-Distillation
git pull --ff-only origin main
export AGENT_DISTILLATION_MODEL_PATH=/mnt/workspace/models/Qwen3.5-0.8B

# Tokenize/check both full datasets; does not load model weights or train.
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --preflight

# Run sequentially: each smoke test uses the same 16 longest paired questions,
# for two optimizer updates. Separate output directories from formal training.
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --smoke
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" shortest --smoke
```

After BOTH smoke tests succeed, run the full experiments sequentially:

```bash
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" normal && \
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" shortest
```

Keep the terminal session alive (or use an existing tmux session). Do not run
the two arms simultaneously on the same GPU. If a smoke test runs out of memory,
stop and adjust the settings consistently for both arms before formal training.

Resume an interrupted formal run from its most recent saved optimizer checkpoint:

```bash
bash scripts/training/run_cot_sft_6400.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --resume
# For shortest, replace normal with shortest.
# For a smoke test, add both --smoke and --resume.
```

Resume requires identical configuration and data. A failure before the first
checkpoint cannot be resumed; use a new output directory with `--output PATH`.
Existing nonempty outputs are never silently overwritten on a fresh run.

## Controlled settings and outputs

- Full chat length cap: 6400, including prompt and special tokens. No truncation,
  no packing, no separate 2048-token assistant cap.
- Native thinking disabled; teacher answers containing thinking markers are
  excluded. System, user, and the empty-thinking generation prefix are masked;
  only the assistant continuation and ending marker contribute to loss.
- NF4 double-quantized QLoRA: rank 64, alpha 128, dropout 0.05.
- Learning rate 2e-4, 2 epochs, batch size 1, accumulation 8, seed 42.
- Gradient checkpointing; BF16 when supported, otherwise FP16.
- Checkpoints every 50 optimizer updates, keeping the last two.
- Output: `training_outputs/Qwen3.5-0.8B/cot_NORMAL_max6400_qlora_seed42`,
  where `NORMAL` is `normal` or `shortest`. Smoke outputs append `_smoke`.
- `final/` contains the LoRA adapter and tokenizer, not standalone base weights.
  Evaluation must load the original base checkpoint plus this adapter.
- `training_manifest.json` records settings, selected IDs and canonical data
  hashes. `training_result.json` records completion and training metrics.

The 188-question validation set is held out and checked for ID overlap. This
script does not evaluate it or select a best checkpoint: evaluate both final
adapters separately with the same generation and grading configuration. MATH500
must remain a test set. Two epochs are an initial controlled setting, not a
claim of optimal hyperparameters.

Local verification: both datasets passed full tokenizer and loss-boundary
preflight with the local Qwen3.5-0.8B tokenizer; unit tests passed. Cloud GPU
training has not been executed by this check.
