import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pytest

from scripts.repair.finetune_verified_repairs import (
    _code_identity,
    _effective_lora_record,
    apply_mode_defaults,
    load_experiment_rows,
)


def _write(path: Path, *, task: str, repair: bool) -> None:
    row = {
        "messages": [
            {"role": "user", "content": task},
            {"role": "assistant", "content": "answer"},
        ]
    }
    if repair:
        row["supervision"] = "last_assistant_only"
        row["target_assistant_turn_index"] = 0
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_mixed_retrain_combines_both_supervision_types() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline = root / "baseline.jsonl"
        repair = root / "repair.jsonl"
        _write(baseline, task="baseline task", repair=False)
        _write(repair, task="repair task", repair=True)

        rows = load_experiment_rows(
            experiment_mode="mixed_retrain",
            baseline_filepath=baseline,
            repair_filepaths=[repair],
            student_lora=None,
        )

        assert [row["supervision"] for row in rows] == [
            "all_assistant_turns",
            "last_assistant_only",
        ]


def test_incremental_repair_requires_s0_and_rejects_baseline() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline = root / "baseline.jsonl"
        repair = root / "repair.jsonl"
        _write(baseline, task="baseline task", repair=False)
        _write(repair, task="repair task", repair=True)

        with pytest.raises(ValueError, match="must not receive --baseline_filepath"):
            load_experiment_rows(
                experiment_mode="incremental_repair",
                baseline_filepath=baseline,
                repair_filepaths=[repair],
                student_lora=root / "s0",
            )
        with pytest.raises(ValueError, match="requires the S0"):
            load_experiment_rows(
                experiment_mode="incremental_repair",
                baseline_filepath=None,
                repair_filepaths=[repair],
                student_lora=None,
            )


def test_mixed_retrain_rejects_s0_and_duplicate_tasks() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        baseline = root / "baseline.jsonl"
        repair = root / "repair.jsonl"
        _write(baseline, task="same task", repair=False)
        _write(repair, task="same task", repair=True)

        with pytest.raises(ValueError, match="must not receive --student_lora"):
            load_experiment_rows(
                experiment_mode="mixed_retrain",
                baseline_filepath=baseline,
                repair_filepaths=[repair],
                student_lora=root / "s0",
            )
        with pytest.raises(ValueError, match="Duplicate task"):
            load_experiment_rows(
                experiment_mode="mixed_retrain",
                baseline_filepath=baseline,
                repair_filepaths=[repair],
                student_lora=None,
            )


def test_incremental_manifest_uses_inherited_adapter_parameters() -> None:
    with TemporaryDirectory() as temp_dir:
        adapter = Path(temp_dir)
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                }
            ),
            encoding="utf-8",
        )
        args = SimpleNamespace(
            experiment_mode="incremental_repair",
            student_lora=str(adapter),
            lora_r=1,
            lora_alpha=2,
            lora_dropout=0.9,
        )

        record = _effective_lora_record(args)

        assert record == {
            "source": "inherited_s0_adapter",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }


@pytest.mark.parametrize(
    ("mode", "epochs", "lr", "save_steps"),
    [
        ("mixed_retrain", 2.0, 2e-4, 25),
        ("incremental_repair", 1.0, 5e-5, 5),
    ],
)
def test_mode_defaults(mode: str, epochs: float, lr: float, save_steps: int) -> None:
    args = SimpleNamespace(
        experiment_mode=mode,
        num_epochs=None,
        lr=None,
        save_steps=None,
    )

    apply_mode_defaults(args)

    assert args.num_epochs == epochs
    assert args.lr == lr
    assert args.save_steps == save_steps


def test_explicit_training_schedule_overrides_are_preserved() -> None:
    args = SimpleNamespace(
        experiment_mode="incremental_repair",
        num_epochs=3.0,
        lr=1e-5,
        save_steps=2,
    )

    apply_mode_defaults(args)

    assert (args.num_epochs, args.lr, args.save_steps) == (3.0, 1e-5, 2)


def test_code_identity_ignores_untracked_status_but_checks_commit_and_diff() -> None:
    record = {
        "commit": "abc123",
        "tracked_diff_sha256": "diff456",
        "working_tree_dirty": True,
        "status_short": ["?? output.jsonl"],
    }

    assert _code_identity(record) == {
        "commit": "abc123",
        "tracked_diff_sha256": "diff456",
    }
