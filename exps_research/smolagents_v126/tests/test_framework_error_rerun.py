import json
import tempfile
import unittest
from pathlib import Path

from scripts.inference.prepare_framework_error_rerun import prepare_rerun_dataset


class FrameworkErrorRerunTest(unittest.TestCase):
    def test_selects_only_known_framework_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "math.json"
            results = root / "results.jsonl"
            output = root / "repair" / "framework_errors.json"
            dataset.write_text(
                json.dumps(
                    {
                        "metadata": {"name": "Math500"},
                        "examples": [
                            {"question": "framework", "answer": "1"},
                            {"question": "model failure", "answer": "2"},
                            {"question": "success", "answer": "3"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "question": "framework",
                    "generated_answer": None,
                    "log_data": None,
                    "error": "Error in generating model output:\n"
                    "string indices must be integers, not 'str'",
                },
                {
                    "question": "model failure",
                    "generated_answer": "fallback",
                    "log_data": {"metadata": {"state": "max_steps_error"}},
                },
                {
                    "question": "success",
                    "generated_answer": "3",
                    "log_data": {"metadata": {"state": "success"}},
                },
            ]
            results.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            count = prepare_rerun_dataset(dataset, results, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(
            [example["question"] for example in payload["examples"]],
            ["framework"],
        )
        self.assertEqual(payload["metadata"]["repair_kind"], "framework_error_only")


if __name__ == "__main__":
    unittest.main()
