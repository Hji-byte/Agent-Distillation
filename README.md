# Math Agent Distillation on smolagents 1.26

This repository distills tool-using mathematical reasoning from a Qwen3.5-27B teacher into a Qwen3.5-0.8B student. The agent runtime is a maintained fork of `smolagents==1.26.0`, rather than the older framework snapshot shipped with the reference paper.

The current pipeline is:

```text
MATH problems
  -> Qwen3.5-27B + CodeAgent teacher
  -> execution and answer grading
  -> trajectory validation / token filtering
  -> 1,646-example SFT dataset
  -> QLoRA fine-tuning of Qwen3.5-0.8B
  -> Math500 agent evaluation
```

The repository also contains an isolated research extension for student-state-aware trajectory repair. It does not alter the baseline teacher-generation or SFT paths.

## Repository layout

- `smolagents-1.26.0-fork/`: vendored and modified smolagents v1.26.0 runtime.
- `exps_research/`: experiment runners, trajectory processing, SFT, evaluation, and repair research.
- `scripts/`: user-facing launchers and dataset utilities.
- `data_processor/math_dataset/`: source math datasets used by the experiments.
- `data_processor/processed/sft/`: final, versioned SFT dataset and its summary.
- `docs/`: project structure and conventions.

Generated logs, checkpoints, virtual environments, model weights, private records, and legacy paper snapshots are intentionally excluded from Git.

## Setup

Python 3.11 is recommended.

```bash
uv sync --python 3.11
```

Copy `.env.example` to `.env`, then add the API endpoint and key required by the selected teacher provider. Never commit `.env`.

For student inference and training, pass the model path explicitly or define `AGENT_DISTILLATION_MODEL_PATH`:

```powershell
$env:AGENT_DISTILLATION_MODEL_PATH = "<local-model-directory>"
```

```bash
export AGENT_DISTILLATION_MODEL_PATH=/path/to/Qwen3.5-0.8B
```

This keeps Windows and Alibaba Cloud PAI DSW paths out of the code.

## Main commands

Generate teacher trajectories:

```powershell
./scripts/inference/run_code_teacher_qwen35_27b_api.ps1
```

Train the student on the final 1,646-example dataset:

```powershell
./exps_research/scripts_train/finetune_sft_agent.ps1
```

```bash
bash exps_research/scripts_train/finetune_sft_agent.sh
```

Run the local Math500 baseline or fine-tuned evaluation:

```powershell
./scripts/inference/run_local_qwen35_baseline.ps1
./scripts/inference/run_local_qwen35_finetuned.ps1 `
  -AdapterPath "<adapter-directory>"
```

Linux/DSW equivalents are available beside the PowerShell launchers.

```bash
bash scripts/inference/run_local_qwen35_baseline.sh /path/to/Qwen3.5-0.8B
bash scripts/inference/run_local_qwen35_finetuned.sh \
  /path/to/adapter /path/to/Qwen3.5-0.8B
```

## Current protocol

Teacher generation uses at most five agent steps and a 600-second per-problem timeout. Each model call starts with a 1,280-token output limit; only a detected truncation is retried with a 2,048-token limit. Training examples are filtered to at most 2,048 tokens per assistant turn and 4,096 tokens for the complete conversation.

The default SFT configuration uses QLoRA, two epochs, learning rate `2e-4`, batch size 1, gradient accumulation 8, and sequence length 4,096. Launch-script arguments and Python defaults are kept aligned.

## Training data

The public training artifact is `data_processor/processed/sft/qwen35_27b_math_medium_hard_1646_v126.jsonl`.

| Source | Accepted examples |
| --- | ---: |
| Original validated Medium + Hard trajectories | 1,576 |
| Medium retry recoveries | 26 |
| Hard retry recoveries | 44 |
| Total | 1,646 |

Five retry trajectories exceeded the configured length limits and were excluded. The merged dataset contains no duplicate problem IDs.

## Tests

```bash
python -m unittest discover -s exps_research/smolagents_v126/tests -v
```

## Upstream and license

The agent runtime is based on [huggingface/smolagents](https://github.com/huggingface/smolagents), tag `v1.26.0`. Project-specific changes are documented in `smolagents-1.26.0-fork/FORK_NOTES.md`; the upstream Apache-2.0 license is retained.

This project was inspired by the experimental setup in *Agent Distillation: Training Smaller Language Models to Reason and Act Like Larger Language Models*. Its older framework snapshot is not used at runtime.
