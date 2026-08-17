import json
import tempfile
import unittest
from pathlib import Path

from exps_research.train_utils.message_utils import (
    get_clean_message_list,
    prepare_sft_messages,
    remove_tool_call_from_messages,
)
from exps_research.train_utils.preprocess import preprocess_logs


class MessageUtilsTest(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {
                "role": "MessageRole.SYSTEM",
                "content": [{"type": "text", "text": "Teacher prompt"}],
            },
            {
                "role": "MessageRole.USER",
                "content": [{"type": "text", "text": "New task:\n2+2?<reference>hidden</reference>"}],
            },
            {
                "role": "MessageRole.ASSISTANT",
                "content": [{"type": "text", "text": "Thought: calculate\nCode: print(2+2)"}],
            },
            {
                "role": "MessageRole.TOOL_CALL",
                "content": [{"type": "text", "text": "Calling tools"}],
            },
            {
                "role": "MessageRole.TOOL_RESPONSE",
                "content": [{"type": "text", "text": "Observation:\n4"}],
            },
            {
                "role": "MessageRole.ASSISTANT",
                "content": [{"type": "text", "text": "final_answer(4)"}],
            },
        ]

    def test_prepares_student_messages_without_mutating_source(self):
        cleaned = prepare_sft_messages(self.messages, system_prompt="Student prompt")

        self.assertEqual(
            [message["role"] for message in cleaned],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(cleaned[0]["content"], "Student prompt")
        self.assertNotIn("<reference>", cleaned[1]["content"])
        self.assertEqual(self.messages[0]["role"], "MessageRole.SYSTEM")

    def test_low_level_helpers_match_paper_text_behavior(self):
        normalized = [
            {"role": "system", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "tool-call", "content": [{"type": "text", "text": "call"}]},
            {"role": "tool-response", "content": [{"type": "text", "text": "result"}]},
        ]
        cleaned = get_clean_message_list(
            remove_tool_call_from_messages(normalized),
            role_conversions={"tool-response": "user"},
            flatten_messages_as_text=True,
        )
        self.assertEqual(cleaned, [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "questionresult"},
        ])

    def test_agent_sft_explicitly_disables_native_thinking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "sample.jsonl"
            data_path.write_text(
                json.dumps({"messages": self.messages}) + "\n",
                encoding="utf-8",
            )

            dataset = preprocess_logs(str(data_path), print_first=False)

        self.assertEqual(
            dataset[0]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_normalizes_text_before_thought_without_dropping_it(self):
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "The result is 4.\n\nThought: I will submit it.\n<code>final_answer(4)</code>",
            },
        ]

        cleaned = prepare_sft_messages(messages)

        self.assertTrue(cleaned[2]["content"].startswith("Thought:"))
        self.assertIn("The result is 4.", cleaned[2]["content"])
        self.assertIn("I will submit it.", cleaned[2]["content"])


if __name__ == "__main__":
    unittest.main()
