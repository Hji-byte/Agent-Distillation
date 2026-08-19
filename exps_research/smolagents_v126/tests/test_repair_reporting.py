import json

from exps_research.repair.reporting import summarize_repair_run


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summarize_repair_run_counts_modes_retries_and_tokens(tmp_path):
    scored = tmp_path / "scored.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        scored,
        [
            {
                "score": 1,
                "input_tokens": 10,
                "output_tokens": 2,
                "log_data": {"metadata": {"state": "max_steps_error"}},
            },
            {
                "score": 0,
                "input_tokens": 20,
                "output_tokens": 4,
                "log_data": {"metadata": {"task_id": "failed-1"}},
            },
            {
                "score": 0,
                "input_tokens": 30,
                "output_tokens": 6,
                "log_data": {"metadata": {"task_id": "failed-2"}},
            },
        ],
    )
    _write_jsonl(
        attempts,
        [
            {
                "repair_id": "failed-1",
                "accepted": True,
                "verification_mode": "student_continuation",
                "continuation_step_count": 2,
                "attempts": [
                    {
                        "teacher_action": {
                            "format_retry_count": 1,
                            "input_tokens": 20,
                            "output_tokens": 10,
                        },
                        "verification": {
                            "trace": [
                                {"source": "teacher_repair"},
                                {
                                    "source": "continuation_policy",
                                    "action": {
                                        "format_retry_count": 1,
                                        "input_tokens": 7,
                                        "output_tokens": 3,
                                    },
                                },
                            ]
                        },
                    }
                ],
            },
            {
                "repair_id": "failed-2",
                "accepted": False,
                "retryable_error": True,
                "attempts": [],
            },
        ],
    )

    summary = summarize_repair_run(scored, attempts)

    assert summary["s0"] == {
        "total": 3,
        "correct": 1,
        "failed": 2,
        "normal_success_correct": 0,
        "max_steps_fallback_correct": 1,
    }
    assert summary["repair"]["processed"] == 2
    assert summary["repair"]["accepted"] == 1
    assert summary["repair"]["retryable_error"] == 1
    assert summary["repair"]["completion_status"] == "incomplete_retryable_errors"
    assert summary["repair"]["remaining_repair_ids"] == ["failed-2"]
    assert summary["repair"]["completion_mode"]["student_continuation"] == 1
    assert summary["repair"]["average_continuation_steps_accepted"] == 2
    assert summary["retries"]["teacher_format_retries"] == 1
    assert summary["retries"]["continuation_format_retries"] == 1
    assert summary["token_usage"]["teacher_input_tokens"] == 20
    assert summary["token_usage"]["continuation_output_tokens"] == 3
    assert summary["token_usage"]["s0_evaluation_input_tokens"] == 60
    assert summary["token_usage"]["s0_evaluation_output_tokens"] == 12


def test_summary_uses_latest_retry_result_and_can_be_complete(tmp_path):
    scored = tmp_path / "scored.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    _write_jsonl(
        scored,
        [
            {
                "score": 0,
                "log_data": {"metadata": {"task_id": "retry-then-pass"}},
            }
        ],
    )
    _write_jsonl(
        attempts,
        [
            {
                "repair_id": "retry-then-pass",
                "accepted": False,
                "retryable_error": True,
                "attempts": [],
            },
            {
                "repair_id": "retry-then-pass",
                "accepted": True,
                "verification_mode": "teacher_terminal",
                "continuation_step_count": 0,
                "attempts": [],
            },
        ],
    )

    summary = summarize_repair_run(scored, attempts)

    assert summary["repair"]["completion_status"] == "complete"
    assert summary["repair"]["processed"] == 1
    assert summary["repair"]["raw_attempt_records"] == 2
    assert summary["repair"]["accepted"] == 1
    assert summary["repair"]["retryable_error"] == 0
