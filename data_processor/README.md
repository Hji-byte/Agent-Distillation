# Data

## `math_dataset/`

Source MATH splits used for teacher generation and evaluation. The project currently uses Medium and Hard subsets for distillation and Math500 for evaluation; AIME, GSM8K, and Olympiad files are retained as optional out-of-distribution evaluations.

## `processed/sft/`

Curated artifacts that are small enough and necessary to reproduce training:

- `qwen35_27b_math_medium_hard_1646_v126.jsonl`: final 1,646-example SFT dataset.
- `qwen35_27b_math_medium_hard_1646_v126.summary.json`: merge provenance and validation counts.

Raw API responses and intermediate trajectories belong under the ignored `logs/` directory. Model weights and checkpoints are never stored here.
