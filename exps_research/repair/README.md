# Verifier-grounded local repair

This directory implements the second-stage experiment without changing the
baseline teacher generation or baseline SFT entry points.

## Data flow

1. Train the baseline student `S0` on the ordinary filtered teacher traces.
2. Run `S0` on a disjoint repair-training split and score every trajectory.
3. Run `scripts/repair/generate_local_repairs.py` on the scored failures.
4. The localizer applies **error-aware backward repair**:
   - explicit format, parsing, and execution failures are tried first;
   - otherwise it starts from the last substantive computation and moves
     backward;
   - a pure `final_answer(variable)` action is deferred for wrong answers.
   - smolagents' post-budget plain-text `provide_final_answer()` fallback is
     never treated as a repairable Thought-and-Code action; when it is wrong,
     localization starts from the last real agent action instead.
5. For each candidate, code actions before the candidate are replayed in a
   fresh restricted executor. Qwen3.5-27B receives exactly the original
   trajectory prefix before that action and generates the next
   `Thought + <code>` action. The failed student action, its resulting error
   or observation, and the failure classification are not sent to the teacher.
6. The repaired action is executed. It may finish the task directly, or the
   local S0 continuation policy finishes the trajectory. The existing
   deterministic Qwen Math Grader verifies the final answer. Only a
   wrong-to-correct intervention is accepted, and the completion mode is
   recorded explicitly.
7. `materialize_repair_sft.py` emits only accepted samples. The rejected
   student action and repair instruction are never placed in the SFT messages.
8. `finetune_verified_repairs.py` masks all prefix tokens with `-100`; only the
   final repaired assistant turn contributes loss. It writes a new `S1`
   adapter and refuses to overwrite the supplied `S0` adapter directory.

## Commands

Create fixed, mutually disjoint smoke (50) and formal (500) repair splits from
the official `EleutherAI/hendrycks_math` train split. This excludes all 2,000
teacher-candidate questions and Math500, records the resolved source revision,
and writes an overlap audit:

```powershell
.\.venv\Scripts\python.exe scripts\repair\prepare_repair_split.py
```

On the GPU server, run or resume the complete 50-question smoke pipeline. The
27B teacher is called through the DashScope-compatible API only for candidate
replacement actions; the trained local S0 is always the continuation policy:

```bash
export AGENT_DISTILLATION_MODEL_PATH=/mnt/workspace/models/Qwen3.5-0.8B
export S0_ADAPTER_PATH=/mnt/workspace/Agent-Distillation/training_outputs/Qwen3.5-0.8B/agent_baseline_2epochs_qlora
bash scripts/repair/run_qwen35_repair_pipeline.sh
```

The project `.env` (or exported environment) must provide
`DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`. Re-running the same command
resumes both the one-attempt S0 evaluation and repair generation.

The S0 repair-split evaluation, Qwen3.5-27B replacement action, and local S0
continuation each default to a 2,048-token single-generation limit. These are
recorded separately in the run manifest.

To run the formal split after smoke validation, pass the formal dataset and
500 samples:

```bash
bash scripts/repair/run_qwen35_repair_pipeline.sh \
  "$S0_ADAPTER_PATH" \
  "$AGENT_DISTILLATION_MODEL_PATH" \
  data_processor/math_dataset/train/math_repair_train_500_seed42.json \
  500
```

Repair artifacts are isolated under a run tag (default `repair-v2`). Set a new
tag for a scientifically distinct configuration, for example
`REPAIR_RUN_TAG=repair-v2-formal`. Each directory contains `run_manifest.json`
with model, LoRA, generation, prompt, dataset-hash, and repair settings, plus
`repair_summary.json` with completion modes, acceptance counts, retries, and
token usage. Resuming refuses to mix settings under an existing tag.
The summary separates answers completed by a normal Agent action from correct
answers produced only by the post-budget max-steps fallback.

Before model loading, the wrapper validates Python imports, CUDA availability,
the base-model config, LoRA config and weights, dataset size, and uniqueness of
the selected questions. The manifest also records the Git revision, tracked
working-tree state, package versions, and hashes for the dataset, prompt,
base-model config, and LoRA files.

For an unrestricted run (`REPAIR_MAX_ENTRIES=-1`, the default), the wrapper
returns success only when every scored S0 failure has a final repair outcome.
Transient API/GPU errors remain retryable, are reported as incomplete, and are
retried by the same command. Repeated records for a retryable question are
collapsed by `repair_id` in the summary, using the newest outcome. A deliberately
limited smoke run reports `partial_by_entry_limit` instead of claiming full
completion.

Generate repairs with the trained local student as continuation policy:

```powershell
.\.venv\Scripts\python.exe scripts\repair\generate_local_repairs.py `
  --input <student_scored_failures.jsonl> `
  --output <repair_attempts.jsonl> `
  --run_tag <run_tag> `
  --run_manifest <run_manifest.json> `
  --continuation_model_type transformers `
  --continuation_model_id <Qwen3.5-0.8B_model_directory> `
  --continuation_lora_path <S0_adapter>
```

For a pipeline smoke test, `--continuation_model_type teacher` reuses the
teacher. This does **not** measure whether the student can recover and should
not be used as the main research result.

Materialize verified targets:

```powershell
.\.venv\Scripts\python.exe scripts\repair\materialize_repair_sft.py `
  --input <repair_attempts.jsonl> `
  --output <verified_repair_sft.jsonl>
```

Continue the baseline adapter into a separate `S1` adapter:

```powershell
.\.venv\Scripts\python.exe scripts\repair\finetune_verified_repairs.py `
  --model_name <Qwen3.5-0.8B_model_directory> `
  --student_lora <S0_adapter> `
  --train_filepath <verified_repair_sft.jsonl> `
  --output_dir <new_S1_adapter>
```

The repair split must not overlap the original teacher-training questions or
Math500. Gold answers are used only by the verifier and are not included in the
teacher context. No additional repair instruction is appended to the original
trajectory prefix.
