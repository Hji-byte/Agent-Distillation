from __future__ import annotations

import argparse
import json
from pathlib import Path


def relative_game_path(game_file: str) -> str:
    marker = "/alfworld/"
    normalized = game_file.replace("\\", "/")
    if marker not in normalized:
        raise ValueError(f"Cannot locate ALFWorld-relative path in: {game_file}")
    return normalized.split(marker, 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename ETO state files to their manifest sample_index"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    index_by_game = {
        task["game_file"]: str(task["sample_index"]) for task in manifest["tasks"]
    }
    planned: list[tuple[Path, Path]] = []
    for source in sorted(args.output_dir.glob("*.json")):
        state = json.loads(source.read_text(encoding="utf-8"))
        relative = relative_game_path(state["meta"]["error"])
        if relative not in index_by_game:
            raise ValueError(f"Output game is outside the manifest: {relative}")
        target = source.with_name(index_by_game[relative] + ".json")
        planned.append((source, target))

    target_names = [target.name for _, target in planned]
    if len(target_names) != len(set(target_names)):
        raise RuntimeError("Multiple outputs correspond to the same manifest game")

    temporary: list[tuple[Path, Path]] = []
    for number, (source, target) in enumerate(planned):
        temp = source.with_name(f".reindex-{number}.json")
        source.rename(temp)
        temporary.append((temp, target))
    for temp, target in temporary:
        temp.rename(target)
        print(f"{temp.name} -> {target.name}")


if __name__ == "__main__":
    main()
