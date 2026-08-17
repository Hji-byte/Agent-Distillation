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
5. For each candidate, code actions before the candidate are replayed in a
   fresh restricted executor. Qwen3.5-27B generates only one replacement
   `Thought + <code>` action.
6. The repaired action is executed, the continuation policy finishes the
   trajectory, and the existing deterministic Qwen Math Grader verifies the
   final answer. Only a wrong-to-correct intervention is accepted.
7. `materialize_repair_sft.py` emits only accepted samples. The rejected
   student action and repair instruction are never placed in the SFT messages.
8. `finetune_verified_repairs.py` masks all prefix tokens with `-100`; only the
   final repaired assistant turn contributes loss. It writes a new `S1`
   adapter and refuses to overwrite the supplied `S0` adapter directory.

## Commands

Generate repairs with the trained local student as continuation policy:

```powershell
.\.venv\Scripts\python.exe scripts\repair\generate_local_repairs.py `
  --input <student_scored_failures.jsonl> `
  --output <repair_attempts.jsonl> `
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
teacher repair prompt.
