from scripts.repair.build_recovery_training_data import (
    build_datasets,
    build_recovery_example,
    build_self_success_example,
)

import json


def test_build_recovery_masks_prefix_and_keeps_terminal_suffix() -> None:
    outcome = {
        "accepted": True,
        "repair_id": "7",
        "failure_kind": "wrong_answer",
        "selected_step_index": 1,
        "selected_attempt_index": 0,
        "experiment_config": {"run_tag": "formal"},
        "attempts": [
            {
                "sft_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "prefix"},
                    {"role": "user", "content": "old observation"},
                    {"role": "assistant", "content": "correction"},
                ],
                "verification": {
                    "correct": True,
                    "verification_mode": "student_continuation",
                    "continuation_step_count": 1,
                    "trace": [
                        {
                            "source": "teacher_repair",
                            "action": {"model_output": "correction"},
                            "execution": {
                                "is_final_answer": False,
                                "observation": "Execution logs:\n42",
                            },
                        },
                        {
                            "source": "continuation_policy",
                            "action": {"model_output": "final_answer"},
                            "execution": {"is_final_answer": True},
                        },
                    ],
                },
            }
        ],
    }

    row = build_recovery_example(outcome)

    assert row["supervision"] == "assistant_suffix"
    assert row["supervised_assistant_start_index"] == 1
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert row["messages"][-2]["content"].startswith("Call id: call_2")
    assert row["messages"][-1]["content"] == "final_answer"


def test_build_self_success_requires_natural_success() -> None:
    row = {
        "score": True,
        "log_data": {
            "metadata": {
                "state": "success",
                "success": True,
                "task_id": 1,
                "trajectory_validation": {"valid": True},
            },
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "Thought: done\n<code>final_answer(42)</code>",
                },
            ],
        },
    }

    example = build_self_success_example(row, source_label="formal")

    assert example["supervision"] == "all_assistant_turns"
    assert example["metadata"]["source"] == "self_success"


def test_dataset_builder_excludes_correct_but_structurally_invalid_success(tmp_path) -> None:
    eligible = tmp_path / "eligible.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    scored = tmp_path / "scored.jsonl"
    eligible.write_text(
        json.dumps({"metadata": {"run_tag": "run", "repair_id": "1"}}) + "\n",
        encoding="utf-8",
    )
    outcome = {
        "accepted": True,
        "repair_id": "1",
        "failure_kind": "wrong_answer",
        "selected_step_index": 0,
        "selected_attempt_index": 0,
        "experiment_config": {"run_tag": "run"},
        "attempts": [
            {
                "sft_messages": [
                    {"role": "user", "content": "repair task"},
                    {"role": "assistant", "content": "fixed"},
                ],
                "verification": {
                    "correct": True,
                    "verification_mode": "teacher_terminal",
                    "continuation_step_count": 0,
                    "trace": [
                        {
                            "action": {"model_output": "fixed"},
                            "execution": {"is_final_answer": True},
                        }
                    ],
                },
            }
        ],
    }
    attempts.write_text(json.dumps(outcome) + "\n", encoding="utf-8")
    invalid_success = {
        "score": True,
        "log_data": {
            "metadata": {
                "state": "success",
                "success": True,
                "trajectory_validation": {"valid": False},
            }
        },
    }
    scored.write_text(json.dumps(invalid_success) + "\n", encoding="utf-8")

    repairs, successes, summary = build_datasets(
        eligible_repairs_path=eligible,
        repair_attempts=[("run", attempts)],
        success_scored=[("success", scored)],
    )

    assert len(repairs) == 1
    assert successes == []
    assert summary["counts"]["self_success_rejections"]["invalid_trajectory"] == 1
