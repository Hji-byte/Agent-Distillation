"""Build disjoint smoke and formal repair splits from official MATH train.

The split is deterministic, excludes every question used as a teacher
candidate and every Math500 evaluation question, and never writes the official
worked solution to the output files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


DATASET_ID = "EleutherAI/hendrycks_math"
DEFAULT_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
FORMAL_QUOTAS = {2: 100, 3: 100, 4: 150, 5: 150}
SMOKE_QUOTAS = {2: 10, 3: 10, 4: 15, 5: 15}


def normalized_question(question: str) -> str:
    return " ".join(str(question).split())


def question_hash(question: str) -> str:
    return hashlib.sha256(normalized_question(question).encode("utf-8")).hexdigest()


def parse_level(value: Any) -> int | None:
    match = re.search(r"[1-5]", str(value))
    if match is None:
        return None
    return int(match.group())


def extract_last_boxed(solution: str) -> str:
    """Extract the final boxed answer, including rare ``\boxed 2`` forms."""
    starts = [solution.rfind("\\boxed"), solution.rfind("\\fbox")]
    marker = max(starts)
    if marker < 0:
        raise ValueError("Official MATH solution has no boxed final answer")
    command = "\\boxed" if starts[0] == marker else "\\fbox"
    tail = solution[marker + len(command) :].lstrip()
    if not tail:
        raise ValueError("Boxed final answer is empty")
    if not tail.startswith("{"):
        answer = tail.split("$", 1)[0].strip().rstrip(".").strip()
        if not answer:
            raise ValueError("Unbraced boxed final answer is empty")
        return answer
    opening = marker + len(command) + (len(solution[marker + len(command) :]) - len(tail))
    depth = 0
    for index in range(opening, len(solution)):
        character = solution[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                answer = solution[opening + 1 : index].strip()
                if not answer:
                    raise ValueError("Boxed final answer is empty")
                return answer
    raise ValueError("Boxed final answer has unbalanced braces")


def load_json_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"{path} does not contain an examples list")
    return examples


def excluded_hashes(paths: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        hashes.update(question_hash(row["question"]) for row in load_json_examples(path))
    return hashes


def load_official_train(revision: str) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(DATASET_ID, revision=revision).sha
    rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        dataset = load_dataset(DATASET_ID, config, split="train", revision=resolved_revision)
        for source_index, row in enumerate(dataset):
            try:
                answer = extract_last_boxed(row["solution"])
            except ValueError:
                answer = None
            rows.append(
                {
                    "question": row["problem"],
                    "answer": answer,
                    "level": parse_level(row["level"]),
                    "type": row["type"],
                    "source_config": config,
                    "source_index": source_index,
                    "source_question_sha256": question_hash(row["problem"]),
                }
            )
    return rows, resolved_revision


def deterministic_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['source_question_sha256']}".encode("utf-8")
        ).hexdigest(),
    )


def stratified_select(
    rows: list[dict[str, Any]],
    quotas: dict[int, int],
    *,
    seed: int,
    reserved_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select exact level quotas while balancing subjects within each level."""
    reserved = reserved_hashes or set()
    selected: list[dict[str, Any]] = []
    subjects = sorted({str(row["type"]) for row in rows})
    for level, target in sorted(quotas.items()):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for subject in subjects:
            candidates = (
                row
                for row in rows
                if row["level"] == level
                and row["type"] == subject
                and row["source_question_sha256"] not in reserved
            )
            buckets[subject] = deterministic_order(candidates, seed + level)

        level_selection: list[dict[str, Any]] = []
        offsets = defaultdict(int)
        round_index = 0
        while len(level_selection) < target:
            made_progress = False
            rotation = round_index % max(len(subjects), 1)
            ordered_subjects = subjects[rotation:] + subjects[:rotation]
            for subject in ordered_subjects:
                offset = offsets[subject]
                if offset >= len(buckets[subject]):
                    continue
                level_selection.append(buckets[subject][offset])
                offsets[subject] += 1
                made_progress = True
                if len(level_selection) == target:
                    break
            if not made_progress:
                available = sum(len(bucket) for bucket in buckets.values())
                raise ValueError(
                    f"Not enough eligible Level {level} examples: need {target}, "
                    f"available {available}"
                )
            round_index += 1
        selected.extend(level_selection)
    return selected


def output_example(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": index,
        "question": row["question"],
        "answer": row["answer"],
        "level": f"Level {row['level']}",
        "type": row["type"],
        "dataset_name": "MATH",
        "split": "train",
        "source_config": row["source_config"],
        "source_index": row["source_index"],
        "source_question_sha256": row["source_question_sha256"],
    }


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "levels": dict(sorted(Counter(f"Level {row['level']}" for row in rows).items())),
        "subjects": dict(sorted(Counter(row["type"] for row in rows).items())),
        "level_by_subject": {
            f"Level {level}": dict(
                sorted(Counter(row["type"] for row in rows if row["level"] == level).items())
            )
            for level in sorted({row["level"] for row in rows})
        },
    }


def write_split(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    purpose: str,
    seed: int,
    revision: str,
) -> None:
    examples = [output_example(row, index) for index, row in enumerate(rows)]
    payload = {
        "metadata": {
            "dataset_info": {
                "name": "MATH",
                "source_dataset": DATASET_ID,
                "source_revision": revision,
                "fold": "train",
                "purpose": purpose,
                "examples_num": len(examples),
                "total_examples": len(examples),
                "seed": seed,
                "contains_official_solutions": False,
                "selection": distribution(rows),
            }
        },
        "examples": examples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as writer:
        writer.write(json.dumps(payload, ensure_ascii=False, indent=4) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def main(args: argparse.Namespace) -> None:
    teacher_paths = [Path(path) for path in args.teacher_candidates]
    math500_path = Path(args.math500)
    official_rows, resolved_revision = load_official_train(args.revision)

    official_hashes = [row["source_question_sha256"] for row in official_rows]
    if len(official_hashes) != 7500:
        raise ValueError(
            "Expected exactly 7,500 official MATH train rows; "
            f"found {len(official_hashes)}"
        )
    unique_official_by_hash: dict[str, dict[str, Any]] = {}
    for row in official_rows:
        unique_official_by_hash.setdefault(row["source_question_sha256"], row)
    unique_official = list(unique_official_by_hash.values())

    teacher_exclusions = excluded_hashes(teacher_paths)
    math500_exclusions = excluded_hashes([math500_path])
    all_exclusions = teacher_exclusions | math500_exclusions
    eligible = [
        row
        for row in unique_official
        if row["source_question_sha256"] not in all_exclusions
        and row["level"] in FORMAL_QUOTAS
        and row["answer"] is not None
    ]

    formal = stratified_select(eligible, FORMAL_QUOTAS, seed=args.seed)
    formal_hashes = {row["source_question_sha256"] for row in formal}
    smoke = stratified_select(
        eligible,
        SMOKE_QUOTAS,
        seed=args.seed,
        reserved_hashes=formal_hashes,
    )
    smoke_hashes = {row["source_question_sha256"] for row in smoke}

    formal_path = Path(args.formal_output)
    smoke_path = Path(args.smoke_output)
    audit_path = Path(args.audit_output)
    write_split(
        formal_path,
        formal,
        purpose="formal_repair_candidate_split",
        seed=args.seed,
        revision=resolved_revision,
    )
    write_split(
        smoke_path,
        smoke,
        purpose="repair_pipeline_smoke_test",
        seed=args.seed,
        revision=resolved_revision,
    )

    audit = {
        "schema_version": "math-repair-split-audit-v1",
        "source_dataset": DATASET_ID,
        "source_revision": resolved_revision,
        "official_train_rows": len(official_rows),
        "official_train_unique_questions": len(set(official_hashes)),
        "official_duplicate_question_rows": len(official_rows) - len(set(official_hashes)),
        "teacher_candidate_files": [repository_path(path) for path in teacher_paths],
        "teacher_candidate_unique_questions": len(teacher_exclusions),
        "teacher_candidates_found_in_official_train": len(
            teacher_exclusions & set(official_hashes)
        ),
        "math500_file": repository_path(math500_path),
        "math500_unique_questions": len(math500_exclusions),
        "math500_questions_found_in_official_train": len(
            math500_exclusions & set(official_hashes)
        ),
        "teacher_math500_overlap": len(teacher_exclusions & math500_exclusions),
        "official_unknown_level_rows": sum(row["level"] is None for row in official_rows),
        "official_unextractable_answer_rows": sum(row["answer"] is None for row in official_rows),
        "eligible_level_2_to_5_after_exclusion": len(eligible),
        "eligible_distribution": distribution(eligible),
        "formal": {
            "path": repository_path(formal_path),
            "count": len(formal),
            "sha256": file_sha256(formal_path),
            "distribution": distribution(formal),
        },
        "smoke": {
            "path": repository_path(smoke_path),
            "count": len(smoke),
            "sha256": file_sha256(smoke_path),
            "distribution": distribution(smoke),
        },
        "overlap_checks": {
            "formal_vs_smoke": len(formal_hashes & smoke_hashes),
            "formal_vs_teacher_candidates": len(formal_hashes & teacher_exclusions),
            "formal_vs_math500": len(formal_hashes & math500_exclusions),
            "smoke_vs_teacher_candidates": len(smoke_hashes & teacher_exclusions),
            "smoke_vs_math500": len(smoke_hashes & math500_exclusions),
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="\n") as writer:
        writer.write(json.dumps(audit, ensure_ascii=False, indent=4) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher_candidates",
        nargs="+",
        default=[
            root / "data_processor/math_dataset/train/math_medium_1000_20250430.json",
            root / "data_processor/math_dataset/train/math_1000_20250414.json",
        ],
    )
    parser.add_argument(
        "--math500",
        default=root / "data_processor/math_dataset/test/math_500_20250414.json",
    )
    parser.add_argument(
        "--formal_output",
        default=root / "data_processor/math_dataset/train/math_repair_train_500_seed42.json",
    )
    parser.add_argument(
        "--smoke_output",
        default=root / "data_processor/math_dataset/train/math_repair_smoke_50_seed42.json",
    )
    parser.add_argument(
        "--audit_output",
        default=root / "data_processor/math_dataset/train/math_repair_split_seed42.audit.json",
    )
    main(parser.parse_args())
