# Research implementation

- `unified_framework/`: experiment configuration, dataset loading, model dispatch, evaluation, and shared utilities.
- `smolagents_v126/`: integration with the vendored smolagents 1.26 fork, including trajectory conversion, answer grading, token validation, and local-model helpers.
- `train_utils/`: SFT dataset and training helpers.
- `repair/`: student-state-aware offline repair experiments, isolated from the baseline.
- `finetune_sft.py`: QLoRA SFT entry point.
- `run_experiment.py`: unified experiment entry point.
- `scripts_train/`: Windows and Linux/PAI DSW training launchers.

The framework source itself lives in `../smolagents-1.26.0-fork`; this directory contains project logic, not a duplicate smolagents package.
