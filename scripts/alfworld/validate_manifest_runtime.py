from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from exps_research.alfworld_eto.manifest_tasks import install_alfworld_manifest_loader
from exps_research.alfworld_eto.optional_imports import (
    install_alfworld_environment_exports,
    install_unused_component_stubs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ETO_ROOT = PROJECT_ROOT / "_local" / "upstream" / "ETO"


def normalized(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify manifest IDs against games returned by ALFWorld reset()"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--eto-root", type=Path, default=DEFAULT_ETO_ROOT)
    parser.add_argument("--max-tasks", type=int, default=50)
    args = parser.parse_args()

    eto_root = args.eto_root.resolve()
    sys.path.insert(0, str(eto_root))
    install_unused_component_stubs()
    install_alfworld_environment_exports()

    install_alfworld_manifest_loader(
        args.manifest, eto_root=eto_root, max_tasks=args.max_tasks
    )
    import eval_agent.tasks as tasks_module

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_root = eto_root / "eval_agent" / "data" / "alfworld"
    expected = {
        task["sample_index"]: normalized(data_root / task["game_file"])
        for task in manifest["tasks"][: args.max_tasks]
    }

    tasks, count = tasks_module.AlfWorldTask.load_tasks("train", 1, -1)
    observed: dict[int, str] = {}
    last_env = None
    for task in tasks:
        actual = normalized(task.game_file)
        wanted = expected.get(task.task_id)
        if wanted != actual:
            raise RuntimeError(
                f"Mismatch for task {task.task_id}: expected {wanted}, got {actual}"
            )
        if task.task_id in observed:
            raise RuntimeError(f"Duplicate task ID returned: {task.task_id}")
        if actual in observed.values():
            raise RuntimeError(f"Duplicate game returned: {actual}")
        observed[task.task_id] = actual
        last_env = task.env

    if last_env is not None:
        last_env.close()
    if count != args.max_tasks or len(observed) != args.max_tasks:
        raise RuntimeError(
            f"Expected {args.max_tasks} tasks, loader={count}, observed={len(observed)}"
        )
    if set(observed) != set(expected):
        raise RuntimeError("Observed manifest IDs are incomplete")

    print(
        json.dumps(
            {
                "status": "ok",
                "expected": args.max_tasks,
                "observed": len(observed),
                "unique_ids": len(set(observed)),
                "unique_games": len(set(observed.values())),
                "id_game_mismatches": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
