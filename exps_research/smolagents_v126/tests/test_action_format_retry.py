import unittest

from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from exps_research.smolagents_v126 import run_result_to_legacy_log_data


class SequencedCodeModel(Model):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, stop_sequences=None, **kwargs):
        self.calls.append(messages)
        return self.outputs.pop(0)


def message(content, input_tokens, output_tokens):
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class ActionFormatRetryTest(unittest.TestCase):
    def test_default_behavior_does_not_require_thought(self):
        model = SequencedCodeModel([message("<code>\nfinal_answer(5)\n</code>", 7, 3)])
        agent = CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0)

        result = agent.run("Return five.", return_full_result=True)
        action_step = next(step for step in result.steps if step.get("model_output") is not None)

        self.assertEqual(result.output, 5)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(action_step["format_retry_count"], 0)

    def test_retries_current_action_and_keeps_only_corrected_output(self):
        model = SequencedCodeModel(
            [
                message("<code>\nfinal_answer(5)\n</code>", 7, 3),
                message("Thought: I should return the computed result.\n<code>\nfinal_answer(5)\n</code>", 13, 7),
            ]
        )
        agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=1,
            verbosity_level=0,
            required_action_prefix="Thought:",
            max_action_format_retries=1,
        )

        result = agent.run("Return five.", return_full_result=True)
        action_step = next(step for step in result.steps if step.get("model_output") is not None)

        self.assertEqual(result.output, 5)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(action_step["format_retry_count"], 1)
        self.assertTrue(action_step["model_output"].startswith("Thought:"))
        self.assertNotIn("<code>\nfinal_answer(5)\n</code>", str(action_step["model_input_messages"]))
        self.assertEqual(action_step["token_usage"]["input_tokens"], 20)
        self.assertEqual(action_step["token_usage"]["output_tokens"], 10)
        self.assertIn("omitted the mandatory", str(model.calls[1]))
        log_data = run_result_to_legacy_log_data(result, agent=agent, task="Return five.")
        self.assertEqual(log_data["trajectory_steps"][0]["format_retry_count"], 1)

    def test_valid_action_is_not_retried(self):
        model = SequencedCodeModel(
            [message("Thought: Return the result.\n<code>\nfinal_answer(5)\n</code>", 11, 4)]
        )
        agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=1,
            verbosity_level=0,
            required_action_prefix="Thought:",
            max_action_format_retries=1,
        )

        result = agent.run("Return five.", return_full_result=True)
        action_step = next(step for step in result.steps if step.get("model_output") is not None)

        self.assertEqual(result.output, 5)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(action_step["format_retry_count"], 0)
        self.assertEqual(action_step["token_usage"]["input_tokens"], 11)
        self.assertEqual(action_step["token_usage"]["output_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
