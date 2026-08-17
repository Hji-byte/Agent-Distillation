import unittest

from exps_research.unified_framework.score_answers import select_attempt_per_question


class ScoreResumeTest(unittest.TestCase):
    def test_strict_benchmark_keeps_first_attempt(self):
        entries = [
            {"question": "q1", "generated_answer": None, "error": "timeout"},
            {"question": "q2", "generated_answer": "first"},
            {"question": "q1", "generated_answer": "later success"},
            {"question": "q2", "generated_answer": "later answer"},
        ]

        selected = select_attempt_per_question(entries, attempt_selection="first")

        self.assertEqual(selected, [entries[0], entries[1]])

    def test_prefers_successful_retry_over_stale_failure(self):
        entries = [
            {"question": "q1", "generated_answer": None, "error": "timeout"},
            {"question": "q2", "generated_answer": "wrong"},
            {"question": "q1", "generated_answer": "correct"},
        ]

        selected = select_attempt_per_question(entries)

        self.assertEqual([entry["question"] for entry in selected], ["q1", "q2"])
        self.assertEqual(selected[0]["generated_answer"], "correct")

    def test_keeps_success_if_a_later_retry_fails(self):
        entries = [
            {"question": "q1", "generated_answer": "answer"},
            {"question": "q1", "generated_answer": None, "error": "timeout"},
        ]

        selected = select_attempt_per_question(entries)

        self.assertEqual(selected, [entries[0]])

    def test_keeps_latest_failure_when_no_attempt_succeeds(self):
        entries = [
            {"question": "q1", "generated_answer": None, "error": "first"},
            {"question": "q1", "generated_answer": None, "error": "second"},
        ]

        selected = select_attempt_per_question(entries)

        self.assertEqual(selected, [entries[1]])


if __name__ == "__main__":
    unittest.main()
