"""Select an additional disjoint repair split from official MATH train."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from prepare_repair_split import (
    CONFIGS,
    DATASET_ID,
    DEFAULT_REVISION,
    distribution,
    excluded_hashes,
    file_sha256,
    load_official_train,
    repository_path,
    stratified_select,
    write_split,
)


EXTENSION_QUOTAS = {2: 90, 3: 90, 4: 135, 5: 135}


def main(args: argparse.Namespace) -> None:
    teacher_paths = [Path(path) for path in args.teacher_candidates]
    previous_paths = [Path(path) for path in args.previous_splits]
    math500_path = Path(args.math500)

    official_rows, resolved_revision = load_official_train(args.revision)
    if len(official_rows) != 7500:
        raise ValueError(
            "Expected exactly 7,500 official MATH train rows; "
            f"found {len(official_rows)}"
        )

    unique_official_by_hash: dict[str, dict[str, Any]] = {}
    for row in official_rows:
        unique_official_by_hash.setdefault(row["source_question_sha256"], row)
    unique_official = list(unique_official_by_hash.values())
    official_hashes = set(unique_official_by_hash)

    teacher_exclusions = excluded_hashes(teacher_paths)
    previous_exclusions = excluded_hashes(previous_paths)
    math500_exclusions = excluded_hashes([math500_path])
    all_exclusions = teacher_exclusions | previous_exclusions | math500_exclusions

    eligible = [
        row
        for row in unique_official
        if row["source_question_sha256"] not in all_exclusions
        and row["level"] in EXTENSION_QUOTAS
        and row["answer"] is not None
    ]
    extension = stratified_select(eligible, EXTENSION_QUOTAS, seed=args.seed)
    extension_hashes = {row["source_question_sha256"] for row in extension}

    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    write_split(
        output_path,
        extension,
        purpose="additional_formal_repair_candidate_split",
        seed=args.seed,
        revision=resolved_revision,
    )

    overlap_checks = {
        "extension_vs_teacher_candidates": len(extension_hashes & teacher_exclusions),
        "extension_vs_previous_550": len(extension_hashes & previous_exclusions),
        "extension_vs_math500": len(extension_hashes & math500_exclusions),
    }
    if any(overlap_checks.values()):
        raise ValueError(f"Disjointness check failed: {overlap_checks}")

    audit = {
        "schema_version": "math-repair-extension-audit-v1",
        "source_dataset": DATASET_ID,
        "source_revision": resolved_revision,
        "official_train_rows": len(official_rows),
        "official_train_unique_questions": len(official_hashes),
        "teacher_candidate_files": [repository_path(path) for path in teacher_paths],
        "teacher_candidate_unique_questions": len(teacher_exclusions),
        "teacher_candidates_found_in_official_train": len(
            teacher_exclusions & official_hashes
        ),
        "previous_split_files": [repository_path(path) for path in previous_paths],
        "previous_split_unique_questions": len(previous_exclusions),
        "previous_questions_found_in_official_train": len(
            previous_exclusions & official_hashes
        ),
        "math500_file": repository_path(math500_path),
        "math500_unique_questions": len(math500_exclusions),
        "math500_questions_found_in_official_train": len(
            math500_exclusions & official_hashes
        ),
        "exclusion_overlap": {
            "teacher_vs_previous_550": len(teacher_exclusions & previous_exclusions),
            "teacher_vs_math500": len(teacher_exclusions & math500_exclusions),
            "previous_550_vs_math500": len(previous_exclusions & math500_exclusions),
        },
        "eligible_level_2_to_5_after_all_exclusions": len(eligible),
        "eligible_distribution": distribution(eligible),
        "extension": {
            "path": repository_path(output_path),
            "count": len(extension),
            "sha256": file_sha256(output_path),
            "distribution": distribution(extension),
        },
        "overlap_checks": overlap_checks,
        "selected_source_configs": dict(
            sorted(Counter(row["source_config"] for row in extension).items())
        ),
        "official_configs": list(CONFIGS),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--teacher_candidates",
        nargs="+",
        default=[
            root / "data_processor/math_dataset/train/math_medium_1000_20250430.json",
            root / "data_processor/math_dataset/train/math_1000_20250414.json",
        ],
    )
    parser.add_argument(
        "--previous_splits",
        nargs="+",
        default=[
            root / "data_processor/math_dataset/train/math_repair_train_500_seed42.json",
            root / "data_processor/math_dataset/train/math_repair_smoke_50_seed42.json",
        ],
    )
    parser.add_argument(
        "--math500",
        default=root / "data_processor/math_dataset/test/math_500_20250414.json",
    )
    parser.add_argument(
        "--output",
        default=root
        / "data_processor/math_dataset/train/math_repair_train_extra_450_seed43.json",
    )
    parser.add_argument(
        "--audit_output",
        default=root
        / "data_processor/math_dataset/train/math_repair_train_extra_450_seed43.audit.json",
    )
    main(parser.parse_args())
