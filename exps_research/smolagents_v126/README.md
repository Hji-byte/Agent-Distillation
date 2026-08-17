# smolagents 1.26 project integration

This package contains the project-specific bridge between the experiment code
and the active smolagents 1.26 fork.

- `trajectory_adapter.py` converts a smolagents 1.26 `RunResult` into the
  project `messages` / `trajectory_steps` / `metadata` structure used by
  filtering and SFT. It also exposes
  `run_result_to_sft_example()` for callers that want the two conversion
  stages in one operation.
- New records use schema `smolagents-v126-native-v1`, retain source-dataset
  fields, and store whole-run plus per-step token usage. Monetary prices are
  deliberately not stored.
- Agent generation uses the native smolagents 1.26 `<code>...</code>`
  protocol through the standard `CodeAgent`.
- `prompts/` contains the paper-compatible, math-only prompt.
- Project adapters and prompts live here. Framework-level validation lives in
  `smolagents-1.26.0-fork/src/smolagents/trajectory_validation.py` and is
  exported by the fork itself.

The old paper fork is retained only in the ignored local archive. It is not
installed by the root project environment or included in the public project.
