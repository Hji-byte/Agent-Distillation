import json
import unittest
from types import SimpleNamespace

from exps_research.smolagents_v126.trajectory_adapter import (
    run_result_to_legacy_log_data,
    run_result_to_sft_example,
)


class TrajectoryAdapterTest(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(
            memory=SimpleNamespace(
                system_prompt=SimpleNamespace(system_prompt="Solve math with Python.")
            ),
            model=SimpleNamespace(model_id="qwen3.7-plus"),
        )
        self.run_result = {
            "output": "\\boxed{4}",
            "state": "success",
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "timing": {"start_time": 1_700_000_000.0, "end_time": 1_700_000_002.0},
            "steps": [
                {"task": "What is 2+2?"},
                {
                    "step_number": 1,
                    "timing": {"start_time": 1.0, "end_time": 1.5, "duration": 0.5},
                    "model_output": "Thought: calculate.\n<code>\nprint(2+2)\n</code>",
                    "code_action": "print(2+2)",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "python_interpreter", "arguments": "print(2+2)"},
                        }
                    ],
                    "observations": "Execution logs:\n4",
                    "error": None,
                    "action_output": None,
                    "token_usage": {"input_tokens": 100, "output_tokens": 50},
                },
                {
                    "step_number": 2,
                    "timing": {"start_time": 1.5, "end_time": 2.0, "duration": 0.5},
                    "model_output": "Thought: report the result.\n<code>\nfinal_answer(r'\\boxed{4}')\n</code>",
                    "code_action": "final_answer(r'\\boxed{4}')",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "python_interpreter", "arguments": "final_answer(r'\\boxed{4}')"},
                        }
                    ],
                    "observations": None,
                    "error": None,
                    "action_output": "\\boxed{4}",
                    "is_final_answer": True,
                },
                {"output": "\\boxed{4}"},
            ],
        }

    def test_converts_core_trajectory_and_metadata(self):
        log_data = run_result_to_legacy_log_data(
            self.run_result,
            agent=self.agent,
            task_id="math-1",
            source_metadata={
                "id": "math-1",
                "dataset_name": "MATH",
                "split": "train",
                "level": "Level 1",
                "type": "Algebra",
            },
        )

        self.assertEqual([m["role"] for m in log_data["messages"]], [
            "system", "user", "assistant", "tool-call", "tool-response", "assistant", "tool-call"
        ])
        self.assertIn("Execution logs:\n4", log_data["messages"][4]["content"][0]["text"])
        self.assertEqual(log_data["metadata"]["final_answer"], "\\boxed{4}")
        self.assertTrue(log_data["metadata"]["success"])
        self.assertEqual(log_data["metadata"]["task_id"], "math-1")
        self.assertNotIn("cost", log_data["metadata"])
        self.assertEqual(log_data["schema_version"], "smolagents-v126-native-v1")
        self.assertEqual(log_data["source"]["level"], "Level 1")
        self.assertEqual(log_data["metadata"]["token_usage"], {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        })
        self.assertEqual(len(log_data["trajectory_steps"]), 2)
        self.assertEqual(log_data["trajectory_steps"][1]["assistant_turn_index"], 1)
        self.assertTrue(log_data["trajectory_steps"][1]["is_final_answer"])
        self.assertEqual(log_data["trajectory_steps"][0]["token_usage"], {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        })
        self.assertEqual(log_data["trajectory_steps"][0]["duration"], 0.5)
        json.dumps(log_data)

    def test_infers_system_prompt_from_model_input(self):
        result = dict(self.run_result)
        result["steps"] = [
            {"task": "What is 2+2?"},
            {
                "model_input_messages": [
                    {"role": "system", "content": [{"type": "text", "text": "System from step"}]}
                ],
                "model_output": "Thought: done",
                "tool_calls": [],
                "observations": None,
                "error": None,
            },
            {"output": "4"},
        ]

        log_data = run_result_to_legacy_log_data(result)
        self.assertEqual(
            log_data["original_memory"]["system_prompt"]["system_prompt"],
            "System from step",
        )

    def test_requires_full_run_result(self):
        with self.assertRaisesRegex(ValueError, "return_full_result=True"):
            run_result_to_legacy_log_data({"output": "4"}, system_prompt="prompt", task="task")

    def test_one_step_sft_conversion(self):
        example = run_result_to_sft_example(
            self.run_result,
            agent=self.agent,
            sft_system_prompt="Student math prompt.",
            source_metadata={"id": "math-1", "level": "Level 1"},
        )

        self.assertEqual(example["messages"][0], {
            "role": "system",
            "content": "Student math prompt.",
        })
        self.assertEqual(
            [message["role"] for message in example["messages"]],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertIn("Observation:\nExecution logs:\n4", example["messages"][3]["content"])
        self.assertEqual(example["metadata"]["final_answer"], "\\boxed{4}")
        self.assertEqual(example["schema_version"], "smolagents-v126-native-v1")
        self.assertEqual(example["source"], {"id": "math-1", "level": "Level 1"})


if __name__ == "__main__":
    unittest.main()
