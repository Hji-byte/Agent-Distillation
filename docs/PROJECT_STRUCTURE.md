# Project structure

## Active execution path

```text
scripts/inference and exps_research/scripts_train
  -> exps_research/unified_framework
  -> exps_research/smolagents_v126
  -> smolagents-1.26.0-fork/src/smolagents
```

`smolagents-1.26.0-fork` is the only smolagents implementation used by the current project. `exps_research/smolagents_v126` is an adapter and data-processing layer; it is not a second copy of the framework.

## Ownership

| Path | Purpose | Commit to GitHub |
| --- | --- | --- |
| `smolagents-1.26.0-fork/` | Modified smolagents v1.26.0 source and focused tests | Yes |
| `exps_research/unified_framework/` | Shared experiment configuration and runners | Yes |
| `exps_research/smolagents_v126/` | Teacher trajectory conversion, grading, validation, and local model integration | Yes |
| `exps_research/repair/` | Isolated student-state-aware repair experiment | Yes |
| `scripts/inference/` | Reproducible teacher and student launchers | Yes |
| `scripts/analysis/` | SFT preprocessing, retry construction, and merging | Yes |
| `data_processor/processed/sft/` | Final reproducible training dataset | Yes |
| `logs/`, `training_outputs/`, `.venv/` | Generated artifacts and local environments | No |
| `_local/`, `record/` | Local archives and conversation records | No |

## Data flow

1. Load Medium or Hard MATH examples from `data_processor/math_dataset`.
2. Generate CodeAgent trajectories with the Qwen3.5 teacher.
3. Execute tool calls and grade the final mathematical answer.
4. Convert successful trajectories into chat messages without changing the teacher's semantic output.
5. Enforce the 2,048-token assistant-turn and 4,096-token conversation limits.
6. Merge validated original and retry data into the versioned SFT JSONL.
7. Fine-tune Qwen3.5-0.8B with QLoRA and evaluate it through the same agent runtime.

## Output convention

Runtime outputs go under ignored directories:

```text
logs/<experiment>/<run-id>/
training_outputs/<run-id>/
```

Only deliberately curated, reproducible artifacts should be copied into `data_processor/processed/` and committed.

## Naming rules

- Use `smolagents-1.26.0-fork` for the modified upstream framework.
- Use `smolagents_v126` only for the project-side adapter package.
- Put executable entry points in `scripts/` or `exps_research/scripts_train/`.
- Put reusable implementation modules in `exps_research/`.
- Include teacher, dataset split, count, and framework version in published dataset filenames.

The old paper implementation based on smolagents 1.13 and raw historical trajectories has been moved to the ignored `_local/legacy_paper/` archive. It is retained locally for reference but is not part of the public project.
