import unittest

from exps_research.repair.sft import (
    materialize_accepted_repairs,
    tokenize_all_assistant_turns,
    tokenize_last_assistant_only,
)


class FakeTokenizer:
    unk_token_id = -1

    def __call__(self, content, *, add_special_tokens):
        return {"input_ids": list(range(len(content)))}

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 100, "<|im_end|>": 101}.get(token, self.unk_token_id)

    def encode(self, text, *, add_special_tokens):
        return {"system": [1], "user": [2], "assistant": [3], "\n": [9]}.get(
            text, [len(text)]
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors, **kwargs):
        ids = []
        for message in messages:
            role_id = {"system": 1, "user": 2, "assistant": 3}[message["role"]]
            ids.extend([100, role_id, 9, len(message["content"]), 101, 9])
        if add_generation_prompt:
            ids.extend([100, 3, 9])
        return ids


class PrefixUnstableQwenTokenizer(FakeTokenizer):
    """Mimic Qwen3.5 changing a turn when it becomes the final assistant."""

    def __init__(self):
        self.template_calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors, **kwargs):
        self.template_calls += 1
        ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
            **kwargs,
        )
        if messages and messages[-1]["role"] == "assistant" and not add_generation_prompt:
            final_header = len(ids) - 6
            ids[final_header + 3 : final_header + 3] = [77, 78]
        return ids


class FakeBatchEncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        ids = super().apply_chat_template(*args, **kwargs)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class RepairSftTest(unittest.TestCase):
    def test_materializes_only_verified_attempt(self):
        accepted = {
            "accepted": True,
            "repair_id": "r1",
            "failure_kind": "wrong_answer",
            "selected_step_index": 0,
            "selected_attempt_index": 0,
            "teacher_model_id": "teacher",
            "continuation_model_id": "student",
            "attempts": [
                {
                    "sft_messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "fixed"},
                    ],
                    "target_assistant_turn_index": 0,
                }
            ],
        }
        examples = materialize_accepted_repairs([accepted, {"accepted": False}])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["supervision"], "last_assistant_only")

    def test_only_final_assistant_tokens_receive_labels(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "fixed"},
        ]
        tokenized = tokenize_last_assistant_only(FakeTokenizer(), messages, max_length=32)
        target_positions = [index for index, label in enumerate(tokenized["labels"]) if label != -100]
        self.assertTrue(target_positions)
        self.assertGreaterEqual(min(target_positions), 15)

    def test_mixed_retrain_supervises_every_assistant_turn(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "step1"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "step2"},
        ]
        tokenized = tokenize_all_assistant_turns(FakeTokenizer(), messages, max_length=64)
        labeled_values = [label for label in tokenized["labels"] if label != -100]
        self.assertIn(len("step1"), labeled_values)
        self.assertIn(len("step2"), labeled_values)

    def test_qwen_prefix_instability_does_not_require_partial_renders(self):
        tokenizer = PrefixUnstableQwenTokenizer()
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "step1"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "step2"},
        ]
        tokenized = tokenize_all_assistant_turns(tokenizer, messages, max_length=64)
        self.assertEqual(tokenizer.template_calls, 1)
        self.assertEqual(
            sum(label != -100 for label in tokenized["labels"]),
            tokenized["supervised_token_count"],
        )

    def test_incremental_repair_masks_earlier_assistant_turns(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "fixed"},
        ]
        tokenized = tokenize_last_assistant_only(FakeTokenizer(), messages, max_length=64)
        labeled_values = [label for label in tokenized["labels"] if label != -100]
        self.assertNotIn(len("old"), labeled_values)
        self.assertIn(len("fixed"), labeled_values)

    def test_refuses_sequence_truncation(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "fixed"},
        ]
        with self.assertRaisesRegex(ValueError, "silently truncated"):
            tokenize_last_assistant_only(FakeTokenizer(), messages, max_length=4)

    def test_refuses_partial_assistant_target(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "fixed"},
        ]
        with self.assertRaisesRegex(ValueError, "max_assistant_tokens"):
            tokenize_last_assistant_only(
                FakeTokenizer(), messages, max_length=32, max_assistant_tokens=1
            )

    def test_accepts_qwen35_style_batch_encoding(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "fixed"},
        ]
        tokenized = tokenize_last_assistant_only(FakeBatchEncodingTokenizer(), messages)
        self.assertGreater(sum(label != -100 for label in tokenized["labels"]), 0)


if __name__ == "__main__":
    unittest.main()
