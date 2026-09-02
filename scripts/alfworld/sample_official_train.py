from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALFWORLD_DATA = (
    PROJECT_ROOT / "_local" / "upstream" / "ETO" / "eval_agent" / "data" / "alfworld"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_processor" / "alfworld_dataset" / "train"

TASK_TYPE_PREFIXES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def task_type_from_game_file(game_file: Path) -> str:
    task_directory = game_file.parent.parent.name
    for prefix in TASK_TYPE_PREFIXES:
        if task_directory.startswith(prefix + "-"):
            return prefix
    raise ValueError(f"Unknown ALFWorld task type: {task_directory}")


def enumerate_official_train_games(alfworld_data: Path) -> list[Path]:
    os.environ["ALFWORLD_DATA"] = str(alfworld_data)
    with (alfworld_data / "base_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # ALFWorld performs its own solvability/task-type filtering here. Sampling
    # raw game.tw-pddl files would include games outside the official split.
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    environment = AlfredTWEnv(config, train_eval="train")
    games = [Path(path).resolve() for path in environment.game_files]
    if len(games) != len(set(games)):
        raise RuntimeError("ALFWorld returned duplicate train game paths")
    return games


def stratified_half(
    games: list[Path], seed: int
) -> tuple[list[Path], list[Path], dict[str, dict[str, int]]]:
    by_type: dict[str, list[Path]] = defaultdict(list)
    for game in games:
        by_type[task_type_from_game_file(game)].append(game)

    if set(by_type) != set(TASK_TYPE_PREFIXES):
        raise RuntimeError(
            f"Expected six ALFWorld task types, found: {sorted(by_type)}"
        )

    rng = random.Random(seed)
    odd_types: list[str] = []
    quotas: dict[str, int] = {}
    for task_type in TASK_TYPE_PREFIXES:
        by_type[task_type].sort(key=lambda path: path.as_posix())
        rng.shuffle(by_type[task_type])
        quotas[task_type] = len(by_type[task_type]) // 2
        if len(by_type[task_type]) % 2:
            odd_types.append(task_type)

    target = len(games) // 2
    extra = target - sum(quotas.values())
    rng.shuffle(odd_types)
    for task_type in odd_types[:extra]:
        quotas[task_type] += 1

    selected: list[Path] = []
    remaining: list[Path] = []
    counts: dict[str, dict[str, int]] = {}
    for task_type in TASK_TYPE_PREFIXES:
        quota = quotas[task_type]
        selected.extend(by_type[task_type][:quota])
        remaining.extend(by_type[task_type][quota:])
        counts[task_type] = {
            "source": len(by_type[task_type]),
            "selected": quota,
            "remaining": len(by_type[task_type]) - quota,
        }

    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining, counts


def make_manifest(
    *,
    games: list[Path],
    source_total: int,
    other_total: int,
    seed: int,
    alfworld_data: Path,
    counts: dict[str, dict[str, int]],
    role: str,
) -> dict[str, object]:
    entries = []
    relative_paths = []
    for index, game in enumerate(games):
        relative = game.relative_to(alfworld_data).as_posix()
        relative_paths.append(relative)
        entries.append(
            {
                "sample_index": index,
                "task_type": task_type_from_game_file(game),
                "game_file": relative,
                "traj_data_file": (game.parent / "traj_data.json")
                .relative_to(alfworld_data)
                .as_posix(),
            }
        )

    digest = hashlib.sha256("\n".join(sorted(relative_paths)).encode()).hexdigest()
    return {
        "schema_version": 1,
        "dataset": "ALFWorld json_2.1.1",
        "split": "train",
        "role": role,
        "sampling": "task-type-stratified deterministic half",
        "seed": seed,
        "source_total": source_total,
        "selected_total": len(games),
        "complement_total": other_total,
        "task_type_counts": counts,
        "game_files_sha256": digest,
        "tasks": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the official ALFWorld train games into reproducible stratified halves"
    )
    parser.add_argument("--alfworld-data", type=Path, default=DEFAULT_ALFWORLD_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    alfworld_data = args.alfworld_data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    games = enumerate_official_train_games(alfworld_data)
    selected, remaining, counts = stratified_half(games, args.seed)

    selected_manifest = make_manifest(
        games=selected,
        source_total=len(games),
        other_total=len(remaining),
        seed=args.seed,
        alfworld_data=alfworld_data,
        counts=counts,
        role="teacher_trajectory_generation",
    )
    remaining_counts = {
        task_type: {
            "source": values["source"],
            "selected": values["remaining"],
            "remaining": values["selected"],
        }
        for task_type, values in counts.items()
    }
    remaining_manifest = make_manifest(
        games=remaining,
        source_total=len(games),
        other_total=len(selected),
        seed=args.seed,
        alfworld_data=alfworld_data,
        counts=remaining_counts,
        role="held_out_for_future_repair",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / f"alfworld_train_teacher_half_seed{args.seed}.json"
    remaining_path = output_dir / f"alfworld_train_remaining_half_seed{args.seed}.json"
    selected_path.write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    remaining_path.write_text(
        json.dumps(remaining_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source_total": len(games),
                "teacher_total": len(selected),
                "remaining_total": len(remaining),
                "seed": args.seed,
                "task_type_counts": counts,
                "teacher_manifest": str(selected_path),
                "remaining_manifest": str(remaining_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
