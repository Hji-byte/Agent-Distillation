import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.merge_framework_rerun_results import merge_framework_rerun


class MergeFrameworkRerunResultsTest(unittest.TestCase):
    def test_replaces_only_matching_framework_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.jsonl"
            rerun = root / "rerun.jsonl"
            output = root / "corrected.jsonl"
            summary = root / "summary.json"
            baseline_records = [
                {"question": "kept", "generated_answer": "1", "score": True},
                {
                    "question": "replace",
                    "generated_answer": None,
                    "log_data": None,
                    "error": "string indices must be integers, not 'str'",
                    "score": False,
                },
            ]
            rerun_record = {
                "question": "replace",
                "generated_answer": "2",
                "log_data": {"metadata": {"state": "success"}},
                "score": True,
            }
            baseline.write_text(
                "\n".join(json.dumps(record) for record in baseline_records) + "\n",
                encoding="utf-8",
            )
            rerun.write_text(json.dumps(rerun_record) + "\n", encoding="utf-8")

            result = merge_framework_rerun(
                baseline,
                rerun,
                output,
                summary,
            )
            merged = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([record["question"] for record in merged], ["kept", "replace"])
        self.assertEqual(merged[1]["generated_answer"], "2")
        self.assertEqual(result["replaced_framework_errors"], 1)
        self.assertEqual(result["correct_answers"], 2)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
