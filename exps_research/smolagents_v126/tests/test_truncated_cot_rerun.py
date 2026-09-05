import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.truncated_cot_rerun import (
    merge_truncated_rerun,
    prepare_truncated_dataset,
)


class TruncatedCotRerunTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "math500.json"
        baseline = root / "baseline.jsonl"
        examples = [
            {"id": 0, "question": "short", "answer": "1"},
            {"id": 1, "question": "truncated-a", "answer": "2"},
            {"id": 2, "question": "truncated-b", "answer": "3"},
        ]
        dataset.write_text(
            json.dumps({"metadata": {"name": "MATH500"}, "examples": examples}),
            encoding="utf-8",
        )
        records = [
            {**examples[0], "generated_answer": "1", "output_tokens": 100},
            {**examples[1], "generated_answer": "wrong", "output_tokens": 4096},
            {**examples[2], "generated_answer": "wrong", "output_tokens": 4200},
        ]
        baseline.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return dataset, baseline

    def test_prepare_preserves_dataset_order_and_selects_limit_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, baseline = self._write_fixture(root)
            output = root / "rerun.json"

            summary = prepare_truncated_dataset(
                dataset, baseline, output, 4096, 8192
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["truncated_questions"], 2)
        self.assertEqual(
            [entry["question"] for entry in payload["examples"]],
            ["truncated-a", "truncated-b"],
        )
        self.assertEqual(payload["metadata"]["source_max_tokens"], 4096)
        self.assertEqual(payload["metadata"]["rerun_max_tokens"], 8192)

    def test_merge_replaces_only_truncated_records_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, baseline = self._write_fixture(root)
            rerun = root / "rerun.jsonl"
            output = root / "merged.jsonl"
            summary_path = root / "summary.json"
            rerun.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question": "truncated-a",
                                "generated_answer": "2",
                                "output_tokens": 5000,
                            }
                        ),
                        json.dumps(
                            {
                                "question": "truncated-b",
                                "generated_answer": "3",
                                "output_tokens": 8192,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = merge_truncated_rerun(
                dataset, baseline, rerun, output, summary_path, 4096, 8192
            )
            merged = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([record["question"] for record in merged], [
            "short",
            "truncated-a",
            "truncated-b",
        ])
        self.assertEqual([record["generated_answer"] for record in merged], [
            "1",
            "2",
            "3",
        ])
        self.assertEqual(summary["replaced_truncated_questions"], 2)
        self.assertEqual(summary["retained_baseline_questions"], 1)
        self.assertEqual(summary["reruns_reaching_new_limit"], 1)

    def test_merge_rejects_partial_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, baseline = self._write_fixture(root)
            rerun = root / "rerun.jsonl"
            rerun.write_text(
                json.dumps(
                    {
                        "question": "truncated-a",
                        "generated_answer": "2",
                        "output_tokens": 5000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing=1"):
                merge_truncated_rerun(
                    dataset,
                    baseline,
                    rerun,
                    root / "merged.jsonl",
                    root / "summary.json",
                    4096,
                    8192,
                )


if __name__ == "__main__":
    unittest.main()
