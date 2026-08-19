import hashlib
import unittest
from types import SimpleNamespace

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from exps_research.repair import RepairConfig, RepairPipeline
from exps_research.repair.messages import (
    REPAIR_PROMPT_VERSION,
    build_teacher_repair_messages,
    repair_prompt_sha256,
)


class SequenceModel(Model):
    def __init__(self, model_id, outputs):
        self.model_id = model_id
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, stop_sequences=None, **kwargs):
        self.calls.append(messages)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=self.outputs.pop(0),
            raw=SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="length" if len(self.calls) == 1 else "stop")]
            ),
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class ConnectionFailingModel(Model):
    model_id = "unavailable-teacher"

    def generate(self, messages, stop_sequences=None, **kwargs):
        raise ConnectionError("temporary API outage")


def failed_two_step_entry():
    return {
        "question": "What is 2 + 3?",
        "true_answer": "5",
        "generated_answer": "4",
        "score": 0,
        "log_data": {
            "metadata": {"state": "success", "task_id": "sample-1"},
            "messages": [
                {"role": "system", "content": "Use Thought and code."},
                {"role": "user", "content": "New task:\nWhat is 2 + 3?"},
                {"role": "assistant", "content": "Thought: compute\n<code>\nx = 4\n</code>"},
                {"role": "tool-call", "content": "Calling tools"},
                {"role": "tool-response", "content": "Observation:\n4"},
                {"role": "assistant", "content": "Thought: answer\n<code>\nfinal_answer(x)\n</code>"},
            ],
            "trajectory_steps": [
                {
                    "assistant_turn_index": 0,
                    "model_output": "Thought: compute\n<code>\nx = 4\n</code>",
                    "code_action": "x = 4",
                    "observations": "4",
                    "error": None,
                    "is_final_answer": False,
                },
                {
                    "assistant_turn_index": 1,
                    "model_output": "Thought: answer\n<code>\nfinal_answer(x)\n</code>",
                    "code_action": "final_answer(x)",
                    "observations": None,
                    "error": None,
                    "is_final_answer": True,
                },
            ],
        },
    }


class RepairPipelineTest(unittest.TestCase):
    def test_teacher_receives_exact_trajectory_prefix_without_repair_diagnostics(self):
        prefix = [
            {"role": "system", "content": "Use Thought and code."},
            {"role": "user", "content": "New task:\nWhat is 2 + 3?"},
        ]

        messages = build_teacher_repair_messages(prefix=prefix)

        self.assertEqual(REPAIR_PROMPT_VERSION, "trajectory-prefix-only-v1")
        self.assertEqual(repair_prompt_sha256(), hashlib.sha256(b"").hexdigest())
        self.assertEqual(len(messages), len(prefix))
        self.assertEqual([str(message.content) for message in messages], [item["content"] for item in prefix])

    def test_repairs_substantive_step_then_verifies_with_continuation_policy(self):
        teacher = SequenceModel("teacher", ["Thought: correct x\n<code>\nx = 5\n</code>"])
        student = SequenceModel("student", ["Thought: submit x\n<code>\nfinal_answer(x)\n</code>"])
        pipeline = RepairPipeline(
            teacher_model=teacher,
            continuation_model=student,
            config=RepairConfig(max_candidates=2, max_continuation_steps=2),
        )

        outcome = pipeline.repair_entry(failed_two_step_entry())

        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["selected_step_index"], 0)
        attempt = outcome["attempts"][outcome["selected_attempt_index"]]
        self.assertTrue(attempt["verification"]["correct"])
        self.assertEqual(attempt["verification"]["final_answer"], "5")
        self.assertEqual(attempt["verification"]["verification_mode"], "student_continuation")
        self.assertEqual(attempt["verification"]["continuation_step_count"], 1)
        self.assertFalse(attempt["original_step_is_final"])
        self.assertFalse(attempt["repaired_step_is_final"])
        self.assertEqual(outcome["verification_mode"], "student_continuation")
        self.assertEqual(outcome["continuation_step_count"], 1)
        self.assertEqual(attempt["sft_messages"][-1]["role"], "assistant")
        self.assertIn("x = 5", attempt["sft_messages"][-1]["content"])
        self.assertNotIn("rejected_action", str(attempt["sft_messages"]))
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(len(student.calls), 1)
        self.assertNotIn("x = 4", str(teacher.calls[0]))
        self.assertNotIn("failure_kind", str(teacher.calls[0]))

    def test_records_teacher_terminal_completion_and_experiment_config(self):
        teacher = SequenceModel(
            "teacher",
            ["Thought: compute directly\n<code>\nfinal_answer(5)\n</code>"],
        )
        pipeline = RepairPipeline(
            teacher_model=teacher,
            continuation_model=SequenceModel("student", []),
            config=RepairConfig(max_candidates=1),
            experiment_config={"run_tag": "repair-v2", "continuation": {"lora_path": "/s0"}},
        )

        outcome = pipeline.repair_entry(failed_two_step_entry())

        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["schema_version"], "local-repair-v2")
        self.assertEqual(outcome["verification_mode"], "teacher_terminal")
        self.assertEqual(outcome["continuation_step_count"], 0)
        self.assertEqual(outcome["experiment_config"]["continuation"]["lora_path"], "/s0")
        attempt = outcome["attempts"][outcome["selected_attempt_index"]]
        self.assertFalse(attempt["original_step_is_final"])
        self.assertTrue(attempt["repaired_step_is_final"])
        self.assertEqual(attempt["verification"]["verification_mode"], "teacher_terminal")

    def test_api_failure_is_retryable_not_a_scientific_rejection(self):
        pipeline = RepairPipeline(
            teacher_model=ConnectionFailingModel(),
            continuation_model=SequenceModel("student", []),
            config=RepairConfig(max_candidates=2),
        )
        outcome = pipeline.repair_entry(failed_two_step_entry())
        self.assertFalse(outcome["accepted"])
        self.assertTrue(outcome["retryable_error"])
        self.assertTrue(outcome["attempts"][0]["retryable_error"])

    def test_teacher_format_failure_preserves_outputs_tokens_and_is_retryable(self):
        teacher = SequenceModel("teacher", ["<code>\nprint(1)", "<code>\nprint(2)"])
        pipeline = RepairPipeline(
            teacher_model=teacher,
            continuation_model=SequenceModel("student", []),
            config=RepairConfig(max_candidates=1, max_format_retries=1),
        )

        outcome = pipeline.repair_entry(failed_two_step_entry())

        self.assertFalse(outcome["accepted"])
        self.assertTrue(outcome["retryable_error"])
        attempt = outcome["attempts"][0]
        self.assertTrue(attempt["retryable_error"])
        failure = attempt["teacher_generation_failure"]
        self.assertEqual(failure["format_retry_count"], 1)
        self.assertEqual(failure["input_tokens"], 20)
        self.assertEqual(failure["output_tokens"], 10)
        self.assertEqual(
            [item["raw_model_output"] for item in failure["generation_attempts"]],
            ["<code>\nprint(1)", "<code>\nprint(2)"],
        )
        self.assertEqual(
            [item["finish_reason"] for item in failure["generation_attempts"]],
            ["length", "stop"],
        )
        self.assertTrue(all(item["closing_tag_restored"] for item in failure["generation_attempts"]))
        self.assertTrue(all(item["model_output"].endswith("</code>") for item in failure["generation_attempts"]))
        self.assertIn("missing a non-empty Thought", failure["generation_attempts"][0]["parse_error"])
        retry_prompt = teacher.calls[1][-1].content[0]["text"]
        self.assertEqual(
            retry_prompt,
            "Your previous action omitted the mandatory non-empty line beginning exactly `Thought:`. "
            "Regenerate only the current assistant action. Start with `Thought: <reasoning>`, then provide "
            "the complete <code>...</code> action. Do not discuss the correction.",
        )


if __name__ == "__main__":
    unittest.main()
