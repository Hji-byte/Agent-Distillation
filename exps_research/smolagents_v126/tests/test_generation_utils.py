import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from exps_research.unified_framework.utils import (
    get_answered_questions,
    prepare_output_path,
)
from exps_research.unified_framework.processors.agent import (
    AgentExperimentProcessor,
    _attempt_summary,
    _is_hard_truncated,
)


class GenerationUtilsTest(unittest.TestCase):
    def test_processor_retries_hard_truncation_once(self):
        def run_result(limit):
            if limit == 1280:
                action = {
                    "step_number": 1,
                    "model_output": "Thought: truncated\n<code>\nfor x in range(10):",
                    "code_action": "for x in range(10):",
                    "observations": None,
                    "error": {"type": "AgentParsingError", "message": "incomplete code"},
                    "is_final_answer": False,
                    "token_usage": {"input_tokens": 100, "output_tokens": 1280},
                }
                return SimpleNamespace(
                    output=None,
                    state="success",
                    steps=[{"task": "2+2?"}, action, {"output": None}],
                    token_usage={"input_tokens": 100, "output_tokens": 1280},
                    timing=None,
                )
            action = {
                "step_number": 1,
                "model_output": "Thought: done\n<code>\nfinal_answer(r'\\boxed{4}')\n</code>",
                "code_action": "final_answer(r'\\boxed{4}')",
                "observations": None,
                "error": None,
                "is_final_answer": True,
                "token_usage": {"input_tokens": 120, "output_tokens": 40},
            }
            return SimpleNamespace(
                output="\\boxed{4}",
                state="success",
                steps=[{"task": "2+2?"}, action, {"output": "\\boxed{4}"}],
                token_usage={"input_tokens": 120, "output_tokens": 40},
                timing=None,
            )

        class FakeAgent:
            def __init__(self, *, model, **kwargs):
                self.model = model
                self.memory = SimpleNamespace(
                    system_prompt=SimpleNamespace(system_prompt="Solve math."),
                    steps=[],
                )

            def run(self, task, return_full_result):
                return run_result(self.model.limit)

        primary_model = SimpleNamespace(model_id="qwen3.7-plus", limit=1280)
        retry_model = SimpleNamespace(model_id="qwen3.7-plus", limit=1600)
        processor = AgentExperimentProcessor(
            {
                "model_type": "openai",
                "model_id": "qwen3.7-plus",
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": 1280,
            }
        )

        with (
            patch(
                "exps_research.unified_framework.processors.agent.CodeAgent",
                FakeAgent,
            ),
            patch(
                "exps_research.unified_framework.processors.agent.setup_model",
                return_value=retry_model,
            ) as setup_retry,
        ):
            result = processor.process_entry(
                {
                    "id": 1,
                    "question": "2+2?",
                    "answer": "4",
                    "dataset_name": "MATH",
                },
                primary_model,
                retry_max_tokens=1600,
            )

        setup_retry.assert_called_once()
        metadata = result["log_data"]["metadata"]
        self.assertTrue(metadata["used_truncation_retry"])
        self.assertEqual(len(metadata["generation_attempts"]), 2)
        self.assertFalse(metadata["generation_attempts"][0]["selected"])
        self.assertTrue(metadata["generation_attempts"][1]["selected"])
        self.assertEqual(result["generated_answer"], "\\boxed{4}")
        self.assertEqual(result["input_tokens"], 220)
        self.assertEqual(result["output_tokens"], 1320)
        self.assertEqual(result["selected_output_tokens"], 40)

    def test_transformers_retry_reuses_loaded_model(self):
        def run_result(limit):
            truncated = limit == 1280
            action = {
                "step_number": 1,
                "model_output": (
                    "Thought: truncated\n<code>\nfor x in range(10):"
                    if truncated
                    else "Thought: done\n<code>\nfinal_answer(r'\\boxed{4}')\n</code>"
                ),
                "code_action": "for x in range(10):" if truncated else "final_answer(r'\\boxed{4}')",
                "observations": None,
                "error": (
                    {"type": "AgentParsingError", "message": "incomplete code"}
                    if truncated
                    else None
                ),
                "is_final_answer": not truncated,
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": limit if truncated else 40,
                },
            }
            output = None if truncated else "\\boxed{4}"
            return SimpleNamespace(
                output=output,
                state="success",
                steps=[{"task": "2+2?"}, action, {"output": output}],
                token_usage=action["token_usage"],
                timing=None,
            )

        class FakeAgent:
            def __init__(self, *, model, **kwargs):
                self.model = model
                self.memory = SimpleNamespace(
                    system_prompt=SimpleNamespace(system_prompt="Solve math."),
                    steps=[],
                )

            def run(self, task, return_full_result):
                return run_result(self.model.kwargs["max_new_tokens"])

        local_model = SimpleNamespace(
            model_id="local-qwen3.5-0.8B",
            kwargs={"max_new_tokens": 1280},
        )
        processor = AgentExperimentProcessor(
            {
                "model_type": "transformers",
                "model_id": "local-qwen3.5-0.8B",
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": 1280,
            }
        )

        with (
            patch(
                "exps_research.unified_framework.processors.agent.CodeAgent",
                FakeAgent,
            ),
            patch(
                "exps_research.unified_framework.processors.agent.setup_model",
            ) as setup_retry,
        ):
            result = processor.process_entry(
                {"id": 1, "question": "2+2?", "answer": "4"},
                local_model,
                retry_max_tokens=2048,
            )

        setup_retry.assert_not_called()
        self.assertEqual(local_model.kwargs["max_new_tokens"], 1280)
        self.assertEqual(result["generated_answer"], "\\boxed{4}")
        self.assertTrue(result["log_data"]["metadata"]["used_truncation_retry"])

    def test_only_retries_malformed_actions_that_hit_the_limit(self):
        log_data = {
            "metadata": {
                "state": "success",
                "trajectory_validation": {
                    "valid": False,
                    "reasons": ["parsing_error"],
                },
                "token_usage": {"input_tokens": 100, "output_tokens": 1280},
            },
            "trajectory_steps": [
                {"token_usage": {"input_tokens": 100, "output_tokens": 1280}}
            ],
        }

        self.assertTrue(_is_hard_truncated(log_data, 1280))
        self.assertFalse(_is_hard_truncated(log_data, 1600))

        log_data["metadata"]["trajectory_validation"]["reasons"] = [
            "non_recoverable_step_error"
        ]
        self.assertFalse(_is_hard_truncated(log_data, 1280))

    def test_attempt_summary_records_retry_selection(self):
        log_data = {
            "metadata": {
                "state": "success",
                "trajectory_validation": {"valid": True, "reasons": []},
                "token_usage": {"input_tokens": 90, "output_tokens": 30},
            },
            "trajectory_steps": [],
        }

        summary = _attempt_summary(log_data, max_tokens=1600, selected=True)

        self.assertEqual(summary["max_tokens"], 1600)
        self.assertTrue(summary["selected"])
        self.assertTrue(summary["valid_structure"])
        self.assertFalse(summary["hard_truncated"])

    def test_agent_resume_only_skips_structured_successes(self):
        failed = {
            "question": "retry me",
            "generated_answer": None,
            "error": "temporary API error",
            "log_data": None,
        }
        legacy = {
            "question": "old protocol",
            "generated_answer": "4",
            "log_data": {"metadata": {"state": "success"}},
        }
        successful = {
            "question": "done",
            "generated_answer": "4",
            "log_data": {
                "metadata": {"state": "success"},
                "trajectory_steps": [],
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "results.jsonl"
            result_file.write_text(
                "\n".join(json.dumps(item) for item in (failed, legacy, successful)) + "\n",
                encoding="utf-8",
            )
            answered = get_answered_questions(
                str(result_file),
                require_successful_agent_run=True,
            )

        self.assertEqual(answered, {"done"})

    def test_output_name_contains_generation_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "train"
            dataset_dir.mkdir()
            dataset_file = dataset_dir / "math.json"
            dataset_file.write_text('{"examples": []}', encoding="utf-8")

            paths = prepare_output_path(
                str(dataset_file),
                "qwen3.7-plus",
                log_folder=str(Path(temp_dir) / "logs"),
                max_steps=5,
                max_tokens=1024,
                retry_max_tokens=1600,
                experiment_type="agent",
                additional_postfix=["code_only", "v126_native"],
            )

        self.assertIn(
            "steps=5_max_tokens=1024_retry_max_tokens=1600",
            paths["output_file"],
        )
        self.assertIn("v126_native", paths["output_file"])


if __name__ == "__main__":
    unittest.main()
