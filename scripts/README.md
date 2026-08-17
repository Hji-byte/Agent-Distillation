# Scripts

The scripts directory contains only active, reproducible entry points.

## `inference/`

- `run_code_teacher_qwen35_27b_api.ps1`: generate Qwen3.5-27B teacher trajectories.
- `run_local_qwen35_baseline.ps1` / `.sh`: evaluate the untrained local student on Math500 through smolagents 1.26.
- `run_local_qwen35_finetuned.ps1` / `.sh`: evaluate the trained adapter on Math500 through the same runtime.

Model locations are accepted as arguments or through `AGENT_DISTILLATION_MODEL_PATH`; machine-specific paths are not committed.

## `analysis/`

- `preprocess_sft_data.py`: convert and filter teacher trajectories for SFT.
- `build_failed_retry_dataset.py`: construct retry inputs from failed generations.
- `merge_sft_data.py`: merge validated original and retry trajectories with duplicate and length checks.

## `repair/`

Launchers for the isolated student-state-aware local-repair research extension. These do not modify baseline trajectory generation or training.
