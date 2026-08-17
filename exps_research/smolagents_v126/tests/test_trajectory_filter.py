import copy
import json
import tempfile
import unittest
from pathlib import Path

from exps_research.smolagents_v126.trajectory_adapter import run_result_to_legacy_log_data
from exps_research.unified_framework.filter_agent_training_data import (
    filter_agent_trajectories,
    inspect_agent_trajectory,
)


def make_run_result():
    return {
        "output": "\\boxed{4}",
        "state": "success",
        "steps": [
            {"task": "What is 2+2?"},
            {
                "step_number": 1,
                "model_output": (
                    "Thought: calculate.\n<code>\nprint(2 + 2)\n</code>"
                ),
                "code_action": "print(2 + 2)",
                "observations": "4",
                "error": None,
                "is_final_answer": False,
            },
            {
                "step_number": 2,
                "model_output": (
                    "Thought: return it.\n<code>\n"
                    "final_answer(r'\\boxed{4}')\n</code>"
                ),
                "code_action": "final_answer(r'\\boxed{4}')",
                "observations": None,
                "error": None,
                "is_final_answer": True,
            },
            {"output": "\\boxed{4}"},
        ],
    }


def make_scored_result(run_result=None):
    return {
        "score": 1,
        "log_data": run_result_to_legacy_log_data(
            run_result or make_run_result(),
            system_prompt="Solve math with Python.",
            task="What is 2+2?",
            model_id="qwen3.7-plus",
        ),
    }


class TrajectoryFilterTest(unittest.TestCase):
    def test_accepts_complete_successful_trajectory(self):
        self.assertTrue(inspect_agent_trajectory(make_scored_result())["valid"])

    def test_keeps_execution_error_and_marks_assistant_turn(self):
        run_result = make_run_result()
        run_result["steps"][1]["error"] = {
            "type": "AgentExecutionError",
            "message": "Import from fractions is not allowed",
        }
        result = make_scored_result(run_result)
        report = inspect_agent_trajectory(result)

        self.assertTrue(report["valid"])
        self.assertEqual(report["execution_error_assistant_turns"], [0])

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample_scored.jsonl"
            source.write_text(json.dumps(result) + "\n", encoding="utf-8")
            stats = filter_agent_trajectories(str(source))
            saved = json.loads(
                (Path(temp_dir) / "sample_filtered.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(stats["valid_entries"], 1)
        self.assertEqual(
            saved["log_data"]["metadata"]["execution_error_assistant_turns"],
            [0],
        )

    def test_rejects_parsing_error(self):
        run_result = make_run_result()
        run_result["steps"][1]["error"] = {
            "type": "AgentParsingError",
            "message": "Could not parse code",
        }
        report = inspect_agent_trajectory(make_scored_result(run_result))
        self.assertFalse(report["valid"])
        self.assertIn("parsing_error", report["reasons"])

    def test_rejects_incomplete_or_non_final_trajectory(self):
        incomplete = make_scored_result()
        incomplete["log_data"]["trajectory_steps"][0]["model_output"] = "Thought: truncated"
        report = inspect_agent_trajectory(incomplete)
        self.assertIn("incomplete_action_format", report["reasons"])

        non_final = make_scored_result()
        non_final["log_data"]["trajectory_steps"][-1]["is_final_answer"] = False
        report = inspect_agent_trajectory(non_final)
        self.assertIn("missing_final_answer", report["reasons"])

    def test_rejects_unsuccessful_run_state(self):
        result = make_scored_result()
        result["log_data"]["metadata"]["state"] = "max_steps_error"
        report = inspect_agent_trajectory(result)
        self.assertFalse(report["valid"])
        self.assertIn("state_not_success", report["reasons"])

    def test_rejects_legacy_unstructured_log(self):
        result = copy.deepcopy(make_scored_result())
        result["log_data"].pop("trajectory_steps")
        report = inspect_agent_trajectory(result)
        self.assertIn("missing_structured_run_result", report["reasons"])

    def test_writes_windows_safe_filtered_data_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluations = Path(temp_dir) / "evaluations"
            evaluations.mkdir()
            source = evaluations / "sample_scored.jsonl"
            source.write_text(json.dumps(make_scored_result()) + "\n", encoding="utf-8")

            filter_agent_trajectories(str(source))

            self.assertTrue(
                (Path(temp_dir) / "filtered_data" / "sample_filtered.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
