"""Keep only counterfactually verified repairs and emit repair SFT JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exps_research.repair.sft import materialize_repair_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="local-repair-v1/v2 attempt JSONL")
    parser.add_argument("--output", required=True, help="local-repair-sft-v1 JSONL")
    args = parser.parse_args()
    count = materialize_repair_jsonl(args.input, args.output)
    print(f"Wrote {count} verified repair SFT examples to {args.output}")
