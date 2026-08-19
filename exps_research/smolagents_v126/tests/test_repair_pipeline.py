import unittest

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from exps_research.repair import RepairConfig, RepairPipeline


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
        self.assertEqual(attempt["sft_messages"][-1]["role"], "assistant")
        self.assertIn("x = 5", attempt["sft_messages"][-1]["content"])
        self.assertNotIn("rejected_action", str(attempt["sft_messages"]))
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(len(student.calls), 1)

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


if __name__ == "__main__":
    unittest.main()
