# ETO ALFWorld compatibility layer

This directory connects local Qwen3.5 checkpoints to the official ETO ALFWorld implementation without editing the upstream Agent framework or prompts.

## Preserved upstream behavior

The following come directly from the pinned ETO commit and are not copied or rewritten here:

- `eval_agent/main.py` interactive loop;
- `AlfWorldTask` and its train/dev/test splits;
- `AlfWorldEnv`, action parser, reward, and 40-step limit;
- ReAct instruction and one-shot ICL example;
- `OpenAILMAgent` and its stop words.

The optional `react-type-2shot` runtime profile leaves those pinned files intact
and injects a separate experimental prompt policy. It routes each game to the
matching ReAct task type, uses the official fixed `_1`, `_0` example order, and
corrects the text command for the PDDL `ToggleObject` action from ETO's
`toggle {obj} {recep}` description to ALFWorld's accepted `use {obj}` command.
Teacher launchers use this profile and a separate `_teacher_api_react2` output
name, so earlier one-shot trajectories are never silently mixed with it.

`integrity.py` verifies the pinned commit and SHA-256 hashes of all protected files before every run. A changed Prompt, Agent loop, environment, or task file causes the run to stop.

## What this layer adds

- an HTTP service that loads Qwen3.5 through the project's existing Transformers path;
- the small legacy API surface expected by ETO's unchanged `OpenAILMAgent`;
- stubs only for unused eager imports of WebShop, ScienceWorld, and FastChat;
- dependency/data setup and reproducible launchers.

The HTTP endpoint is local (`127.0.0.1`): it does not call OpenAI or send trajectories off-machine.
For Qwen3.5 generation, the endpoint explicitly stops on the chat tokenizer's
`<|im_end|>` token. This prevents a fine-tuned model from continuing past its
assistant response into serialized `user` role markers.

## Windows Docker (recommended for this computer)

ALFWorld/TextWorld runs inside Linux while the repository and outputs stay in
the existing Windows project directory. Docker does not change ETO's Prompt or
Agent loop.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/alfworld/docker_alfworld.ps1 build
powershell -ExecutionPolicy Bypass -File scripts/alfworld/docker_alfworld.ps1 setup
```

For a five-task teacher API smoke run, set the key only in the current terminal:

```powershell
$env:ALFWORLD_API_KEY="your-key"
powershell -ExecutionPolicy Bypass -File scripts/alfworld/docker_alfworld.ps1 teacher-smoke
```

The default API base is DashScope's OpenAI-compatible endpoint. Override
`ALFWORLD_API_BASE` and `ALFWORLD_TEACHER_MODEL` when using another compatible
provider. The key is passed through the container environment and written to
an ephemeral runtime config; it is not committed and the config is removed
when the process exits.

Teacher generation defaults to the deterministic, task-type-stratified half
stored at `data_processor/alfworld_dataset/train/alfworld_train_teacher_half_seed42.json`.
The complementary half is kept separately for later student/repair sampling.
This selection adapter changes only which official train games are registered;
the protected ETO Prompt, Agent loop, and ALFWorld environment remain unchanged.

The reusable five-task ReAct 2-shot pilot is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/alfworld/docker_alfworld.ps1 teacher-five
```

It writes to the formal `_teacher_api_react2` experiment directory. A later
`teacher-full` run loads the same manifest and skips those exact sample IDs.

## Setup

```bash
bash scripts/alfworld/setup_eto_alfworld.sh
```

This pins ETO to commit `a2fc5da38f8d00cfaf3f9b6370d586eebaf72904`, installs ALFWorld-only dependencies, downloads the official ETO ALFWorld game data, and verifies protected files.

When expert trajectories are needed for ALFWorld distillation, download ETO's official training package as well:

```bash
bash scripts/alfworld/setup_eto_alfworld.sh training
```

The default `eval` mode downloads only the environment assets. The optional `training` mode additionally extracts ETO's official `data/alfworld_sft.json`; neither mode edits protected Prompt or Agent files.

## Run Base Qwen3.5-0.8B

Terminal 1:

```bash
bash scripts/alfworld/serve_qwen35_local.sh /mnt/workspace/models/Qwen3.5-0.8B
```

Terminal 2, five-task smoke run:

```bash
bash scripts/alfworld/run_eto_alfworld.sh test smoke base
```

Full 134-task unseen test:

```bash
bash scripts/alfworld/run_eto_alfworld.sh test full base
```

The split is included in the experiment name, so `dev` (Valid Seen) and
`test` (Valid Unseen) are saved separately. The third launcher argument is a
model tag and defaults to `base`; use a distinct tag such as `sft` for every
checkpoint being compared. Split and model tag are both included in the
experiment name, preventing Base and fine-tuned results from being resumed or
overwritten as one run.

To evaluate an adapter while keeping the same ETO framework and Prompt, pass it as the second server argument:

```bash
bash scripts/alfworld/serve_qwen35_local.sh \
  /mnt/workspace/models/Qwen3.5-0.8B \
  /mnt/workspace/Agent-Distillation/training_outputs/Qwen3.5-0.8B/alfworld_baseline_2epochs_alfworld_teacher_sft_max6400_qlora
```

Then evaluate the adapter under its own result tag:

```bash
bash scripts/alfworld/run_eto_alfworld.sh test smoke sft
bash scripts/alfworld/run_eto_alfworld.sh test full sft
bash scripts/alfworld/run_eto_alfworld.sh dev full sft
```

ETO writes its original per-task state JSON files under `_local/upstream/ETO/outputs/`. These are kept separate from the Math500 results until an ALFWorld run is complete and deliberately curated.
