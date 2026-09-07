# Paired CoT QLoRA: 4096 version

Use `run_cot_sft_4096.sh` for the v5 dataset: 1887 paired questions per arm,
188 held-out validation questions. The original 6400 entry point still selects
v4 (1940 pairs); its defaults are unchanged. Never resume a 6400 run as 4096.

Full chat length is at most 4096, not an assistant-only cap. There is no
truncation, packing, or separate 2048-token answer limit. Both runs initialize
from the same original base checkpoint, not from each other's adapter.

## Cloud commands

```bash
cd /mnt/workspace/Agent-Distillation
git pull --ff-only origin main
export AGENT_DISTILLATION_MODEL_PATH=/mnt/workspace/models/Qwen3.5-0.8B

# Checks both datasets without loading model weights or creating output files.
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --preflight

# Two optimizer updates per arm. Run sequentially, not concurrently.
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --smoke && \
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" shortest --smoke
```

Only after BOTH smoke tests succeed:

```bash
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" normal && \
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" shortest
```

The terminal must stay alive; use an existing tmux session if needed.
Each smoke run uses the same 16 pairs ranked by their maximum length across
the two arms. Smoke outputs are separate and are not used to initialize the
formal experiments. Passing a smoke test is not a guarantee against all OOMs.

## Settings, outputs, resume

QLoRA NF4 double quantization, rank64 / alpha128 / dropout0.05; learning rate
2e-4, 2 epochs, batch1, gradient accumulation8, seed42, gradient checkpointing.
BF16 on the A10. Other than dataset/length and allocator configuration, the
training method is unchanged; no new loss implementation is introduced.
The wrapper defaults to expandable allocator segments to reduce fragmentation,
not to guarantee sufficient memory. No new dependencies or vLLM are required
relative to the 6400 trainer's working environment.

Outputs:

- `training_outputs/Qwen3.5-0.8B/cot_normal_max4096_qlora_seed42/final`
- `training_outputs/Qwen3.5-0.8B/cot_shortest_max4096_qlora_seed42/final`

These contain adapters and tokenizer, not standalone base weights. Smoke runs
append `_smoke` to the parent directory. Old 6400 outputs remain untouched.
Training only supervises the assistant continuation and ending marker.
Validation generation and MATH500 evaluation are separate, not part of training.

```bash
# Resume an interrupted FORMAL normal run; substitute shortest for the other arm.
bash scripts/training/run_cot_sft_4096.sh "$AGENT_DISTILLATION_MODEL_PATH" normal --resume
```

Resume requires identical script/configuration/data and an optimizer checkpoint
(saved every50 updates, retaining two). If it fails before the first checkpoint,
use a new output directory via `--output PATH` instead of `--resume`; never
delete old results blindly. Re-running a completed smoke test likewise needs a
new output directory. Full runs start from base, without `--resume`.

Local verification covers tokenizer lengths, masks and unit tests. A10 GPU
forward/backward memory has not been tested locally. If OOM persists, stop and
inspect the traceback rather than silently dropping more samples.
