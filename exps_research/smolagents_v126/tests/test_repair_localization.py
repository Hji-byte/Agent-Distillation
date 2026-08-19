import unittest

from exps_research.repair.localization import classify_failure, error_aware_backward_candidates


def step(code, *, turn, final=False, error=None, output=None):
    return {
        "assistant_turn_index": turn,
        "model_output": output or f"Thought: action {turn}\n<code>\n{code}\n</code>",
        "code_action": code,
        "observations": "ok",
        "error": error,
        "is_final_answer": final,
    }


class RepairLocalizationTest(unittest.TestCase):
    def test_wrong_answer_defers_pure_final_submission(self):
        entry = {
            "score": 0,
            "log_data": {
                "metadata": {"state": "success"},
                "trajectory_steps": [
                    step("x = 4", turn=0),
                    step("final_answer(x)", turn=1, final=True),
                ],
            },
        }
        candidates = error_aware_backward_candidates(entry)
        self.assertEqual(classify_failure(entry), "wrong_answer")
        self.assertEqual([candidate.step_index for candidate in candidates], [0, 1])
        self.assertEqual(candidates[-1].reason, "answer_submission_fallback")

    def test_explicit_execution_error_is_repaired_first(self):
        entry = {
            "score": 0,
            "log_data": {
                "metadata": {"state": "max_steps_error"},
                "trajectory_steps": [
                    step("x = 4", turn=0),
                    step(
                        "y = missing + 1",
                        turn=1,
                        error={"type": "AgentExecutionError", "message": "execution failed"},
                    ),
                    step("final_answer(0)", turn=2, final=True),
                ],
            },
        }
        candidates = error_aware_backward_candidates(entry)
        self.assertEqual(candidates[0].step_index, 1)
        self.assertEqual(candidates[0].failure_kind, "execution_error")
        self.assertEqual([candidate.step_index for candidate in candidates], [1, 0])

    def test_incomplete_action_is_detected_structurally(self):
        entry = {
            "score": 0,
            "log_data": {
                "metadata": {"state": "max_steps_error"},
                "trajectory_steps": [
                    step("", turn=0, output="Thought: unfinished"),
                ],
            },
        }
        self.assertEqual(classify_failure(entry), "incomplete_action_format")
        self.assertEqual(error_aware_backward_candidates(entry)[0].step_index, 0)

    def test_forced_max_steps_fallback_is_not_a_repair_candidate(self):
        entry = {
            "score": 0,
            "log_data": {
                "metadata": {"state": "max_steps_error"},
                "trajectory_steps": [
                    step("x = 4", turn=0),
                    step("y = x + 1", turn=1),
                    {
                        "assistant_turn_index": None,
                        "model_output": None,
                        "code_action": None,
                        "observations": None,
                        "error": {
                            "type": "AgentMaxStepsError",
                            "message": "Reached max steps.",
                        },
                        "is_final_answer": False,
                    },
                ],
            },
        }

        candidates = error_aware_backward_candidates(entry)

        self.assertEqual(classify_failure(entry), "max_steps")
        self.assertEqual([candidate.step_index for candidate in candidates], [1, 0])
        self.assertNotIn(2, [candidate.step_index for candidate in candidates])


if __name__ == "__main__":
    unittest.main()
