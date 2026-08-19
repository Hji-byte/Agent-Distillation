"""Write a compact JSON summary for one repair run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.repair.reporting import summarize_repair_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", required=True)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_entries", type=int, default=-1)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    summary = summarize_repair_run(args.scored, args.attempts, max_entries=args.max_entries)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote repair summary to {target}")
    if args.require_complete and summary["repair"]["completion_status"] != "complete":
        raise SystemExit(3)
