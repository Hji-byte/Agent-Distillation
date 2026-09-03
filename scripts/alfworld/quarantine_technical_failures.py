from __future__ import annotations

import argparse
import json
from pathlib import Path


TECHNICAL_FAILURE_REASON = "exceeding maximum input length"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move ETO agent-call failures aside so they can be retried"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("archive_dir", type=Path)
    args = parser.parse_args()

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    kept = 0
    for source in sorted(args.output_dir.glob("*.json")):
        if not source.stem.isdigit():
            continue
        state = json.loads(source.read_text(encoding="utf-8"))
        metadata = state["metadata"]
        reason = metadata.get("terminate_reason")
        if reason != TECHNICAL_FAILURE_REASON:
            kept += 1
            continue
        target = args.archive_dir / source.name
        if target.exists():
            raise FileExistsError(f"Archive target already exists: {target}")
        source.rename(target)
        moved += 1

    print(json.dumps({"moved": moved, "kept": kept, "archive": str(args.archive_dir)}))


if __name__ == "__main__":
    main()
