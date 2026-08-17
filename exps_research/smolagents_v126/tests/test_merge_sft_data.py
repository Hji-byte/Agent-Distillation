import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.merge_sft_data import merge_sft_files


def _row(task: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": task},
            {"role": "assistant", "content": answer},
        ]
    }


class MergeSftDataTest(unittest.TestCase):
    def test_preserves_source_order_and_deduplicates_by_task(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base.jsonl"
            retry = root / "retry.jsonl"
            output = root / "merged.jsonl"
            base.write_text(
                "\n".join(json.dumps(row) for row in (_row("q1", "a1"), _row("q2", "base"))) + "\n",
                encoding="utf-8",
            )
            retry.write_text(
                "\n".join(json.dumps(row) for row in (_row("q2", "retry"), _row("q3", "a3"))) + "\n",
                encoding="utf-8",
            )

            summary = merge_sft_files(
                [("base", base), ("retry", retry)],
                output,
                max_length=4096,
                max_assistant_tokens=2048,
                tokenizer_path="tokenizer",
            )

            merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["messages"][0]["content"] for row in merged], ["q1", "q2", "q3"])
            self.assertEqual(merged[1]["messages"][1]["content"], "base")
            self.assertEqual(summary["total_kept"], 3)
            self.assertEqual(summary["total_duplicate_tasks"], 1)
            self.assertTrue(output.with_suffix(".summary.json").exists())


if __name__ == "__main__":
    unittest.main()
