import unittest

from smolagents import CodeAgent
from smolagents.trajectory_validation import validate_run_result_for_sft


def make_steps():
    return [
        {"task": "What is 2+2?"},
        {
            "step_number": 1,
            "model_output": "Thought: calculate.\n<code>\nprint(2 + 2)\n</code>",
            "error": None,
            "is_final_answer": False,
        },
        {
            "step_number": 2,
            "model_output": (
                "Thought: return it.\n<code>\n"
                "final_answer(r'\\boxed{4}')\n</code>"
            ),
            "error": None,
            "is_final_answer": True,
        },
    ]


class TrajectoryValidationTest(unittest.TestCase):
    def test_code_agent_uses_native_protocol(self):
        model = type("ModelStub", (), {"model_id": "stub"})()
        agent = CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0)
        self.addCleanup(agent.cleanup)
        self.assertEqual(agent.code_block_tags, ("<code>", "</code>"))

    def test_accepts_complete_successful_run(self):
        report = validate_run_result_for_sft({"state": "success", "steps": make_steps()})
        self.assertTrue(report.valid)

    def test_keeps_execution_error_and_marks_turn(self):
        steps = make_steps()
        steps[1]["error"] = {
            "type": "AgentExecutionError",
            "message": "Import is not allowed",
        }
        report = validate_run_result_for_sft({"state": "success", "steps": steps})
        self.assertTrue(report.valid)
        self.assertEqual(report.execution_error_assistant_turns, [0])

    def test_rejects_parsing_error(self):
        steps = make_steps()
        steps[1]["error"] = {
            "type": "AgentParsingError",
            "message": "Could not parse code",
        }
        report = validate_run_result_for_sft({"state": "success", "steps": steps})
        self.assertFalse(report.valid)
        self.assertIn("parsing_error", report.reasons)

    def test_rejects_incomplete_non_final_or_failed_run(self):
        steps = make_steps()
        steps[0 + 1]["model_output"] = "Thought: truncated"
        steps[-1]["is_final_answer"] = False
        report = validate_run_result_for_sft({"state": "max_steps_error", "steps": steps})
        self.assertFalse(report.valid)
        self.assertIn("state_not_success", report.reasons)
        self.assertIn("incomplete_action_format", report.reasons)
        self.assertIn("missing_final_answer", report.reasons)


if __name__ == "__main__":
    unittest.main()
