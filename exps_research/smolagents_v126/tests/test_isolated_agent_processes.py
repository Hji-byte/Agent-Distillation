import json
import tempfile
import time
import unittest
from pathlib import Path

from exps_research.smolagents_v126.tests.isolated_worker_fixture import (
    IsolatedWorkerFixtureProcessor,
)
from exps_research.unified_framework.utils import get_answered_questions


class IsolatedAgentProcessesTest(unittest.TestCase):
    def test_hung_question_is_killed_without_blocking_other_questions(self):
        processor = IsolatedWorkerFixtureProcessor({"model_id": "fixture"})
        entries = [
            {"id": 1, "question": "first", "answer": "1"},
            {"id": 2, "question": "hang", "answer": "2", "behavior": "hang"},
            {"id": 3, "question": "last", "answer": "3"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.jsonl"
            started_at = time.monotonic()
            results = processor.process_dataset(
                entries,
                output_file=str(output_file),
                max_workers=2,
                isolate_agent_processes=True,
                question_timeout_seconds=4,
            )
            elapsed = time.monotonic() - started_at

            self.assertLess(elapsed, 15)
            self.assertEqual(len(results), 3)
            by_question = {result["question"]: result for result in results}
            self.assertEqual(by_question["first"]["generated_answer"], "1")
            self.assertEqual(by_question["last"]["generated_answer"], "3")
            self.assertEqual(
                by_question["hang"]["log_data"]["metadata"]["state"],
                "execution_timeout",
            )
            self.assertEqual(
                by_question["hang"]["log_data"]["metadata"]["timeout_seconds"],
                4,
            )

            rows = [
                json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                get_answered_questions(
                    str(output_file),
                    require_successful_agent_run=True,
                ),
                {"first", "last"},
            )

    def test_child_exception_is_recorded_and_remains_retryable(self):
        processor = IsolatedWorkerFixtureProcessor({"model_id": "fixture"})
        entries = [{"question": "broken", "behavior": "error"}]

        results = processor.process_dataset(
            entries,
            max_workers=1,
            isolate_agent_processes=True,
            question_timeout_seconds=4,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("fixture failure", result["error"])
        self.assertEqual(
            result["log_data"]["metadata"]["state"],
            "child_process_error",
        )


if __name__ == "__main__":
    unittest.main()
