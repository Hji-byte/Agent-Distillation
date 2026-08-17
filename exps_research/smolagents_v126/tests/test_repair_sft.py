import unittest

from exps_research.repair.sft import materialize_accepted_repairs, tokenize_last_assistant_only


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors, **kwargs):
        ids = []
        for message in messages:
            role_id = {"system": 1, "user": 2, "assistant": 3}[message["role"]]
            ids.extend([role_id, len(message["content"])])
        if add_generation_prompt:
            ids.append(3)
        else:
            ids.append(9)
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
        self.assertGreaterEqual(min(target_positions), 5)

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
