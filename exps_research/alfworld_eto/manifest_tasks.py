from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def _load_manifest(manifest_path: Path, data_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "ALFWorld json_2.1.1":
        raise ValueError(f"Unsupported task manifest dataset: {manifest.get('dataset')}")
    if manifest.get("split") != "train":
        raise ValueError("The ALFWorld teacher manifest must use the train split")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != manifest.get("selected_total"):
        raise ValueError("Manifest task count does not match selected_total")

    relative_paths = []
    sample_indices = []
    for task in tasks:
        relative = Path(task["game_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe game path in manifest: {relative}")
        game_file = data_root / relative
        if not game_file.is_file():
            raise FileNotFoundError(f"Manifest game is missing: {game_file}")
        relative_paths.append(relative.as_posix())
        sample_indices.append(task["sample_index"])

    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Manifest contains duplicate game paths")
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("Manifest contains duplicate sample indices")

    digest = hashlib.sha256("\n".join(sorted(relative_paths)).encode()).hexdigest()
    if digest != manifest.get("game_files_sha256"):
        raise ValueError("Manifest game path checksum does not match")
    return tasks


def _select_part(
    tasks: list[dict[str, Any]], part_num: int, part_idx: int
) -> list[dict[str, Any]]:
    if part_num < 1:
        raise ValueError("part_num must be at least 1")
    if part_num == 1:
        return tasks
    if not 0 <= part_idx < part_num:
        raise ValueError("part_idx must be in [0, part_num) when sharding")

    part_sizes = [len(tasks) // part_num] * part_num
    part_sizes[-1] += len(tasks) % part_num
    start = sum(part_sizes[:part_idx])
    return tasks[start : start + part_sizes[part_idx]]


def install_alfworld_manifest_loader(
    manifest_path: Path, *, eto_root: Path, max_tasks: int | None = None
) -> dict[str, Any]:
    """Select exact train games while preserving ETO's task/env/agent classes."""
    import alfworld
    import eval_agent.tasks as tasks_module

    data_root = (eto_root / "eval_agent" / "data" / "alfworld").resolve()
    all_entries = _load_manifest(manifest_path.resolve(), data_root)
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        all_entries = all_entries[:max_tasks]
    task_class = tasks_module.AlfWorldTask

    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1):
        if split != "train":
            raise ValueError("A train manifest can only be used with --split train")
        entries = _select_part(all_entries, part_num, part_idx)

        os.environ["ALFWORLD_DATA"] = str(data_root)
        with (data_root / "base_config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        environment = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval="train"
        )
        environment.game_files = [
            str((data_root / entry["game_file"]).resolve()) for entry in entries
        ]
        entries_by_game_file = {
            str((data_root / entry["game_file"]).resolve()): entry for entry in entries
        }
        env = environment.init_env(batch_size=1)

        def generator():
            for _ in entries:
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = str(Path(info["extra.gamefile"][0]).resolve())
                entry = entries_by_game_file.get(game_file)
                if entry is None:
                    raise RuntimeError(
                        f"ALFWorld reset returned a game outside the manifest: {game_file}"
                    )
                yield cls(
                    task_id=entry["sample_index"],
                    game_file=game_file,
                    env=env,
                    obs=obs,
                )

        return generator(), len(entries)

    task_class.load_tasks = classmethod(load_tasks)
    return {
        "manifest": str(manifest_path.resolve()),
        "task_count": len(all_entries),
        "game_files_sha256": hashlib.sha256(
            "\n".join(sorted(entry["game_file"] for entry in all_entries)).encode()
        ).hexdigest(),
    }
