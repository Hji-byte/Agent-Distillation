from __future__ import annotations

import sys
import json
import threading
import types
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from exps_research.alfworld_eto.integrity import EXPECTED_COMMIT, sha256_file, verify_upstream
from exps_research.alfworld_eto.legacy_openai import install_legacy_openai_module
from exps_research.alfworld_eto.openai_compat_server import (
    DEFAULT_STOP,
    QwenTransformersBackend,
    make_handler,
    normalize_stops,
    resolve_chat_termination_token_ids,
)
from exps_research.alfworld_eto.optional_imports import install_unused_component_stubs
from exps_research.alfworld_eto.run_upstream import write_runtime_model_config
from exps_research.alfworld_eto.react_prompting import (
    REACT_EXAMPLE_INDICES,
    correct_textworld_action_instruction,
    load_react_two_shot,
    prompt_type_for_game,
    split_rendered_prompt_into_system_user,
)
from exps_research.alfworld_eto.trajectory_serialization import (
    SCHEMA_VERSION,
    install_normalized_trajectory_serialization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_normalize_stops_preserves_eto_stop_words():
    assert normalize_stops(None) == DEFAULT_STOP
    assert normalize_stops("END") == ["END"]
    assert normalize_stops(["A", "B"]) == ["A", "B"]
    with pytest.raises(ValueError):
        normalize_stops([1])


def test_local_student_and_teacher_use_same_per_step_output_budget():
    config_path = (
        PROJECT_ROOT
        / "exps_research"
        / "alfworld_eto"
        / "configs"
        / "model"
        / "qwen35_local_openai.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["config"]["max_tokens"] == 1024


def test_local_backend_stops_at_chat_tokenizer_eos():
    class FakeTokenizer:
        eos_token_id = 248046  # Qwen3.5 <|im_end|>
        pad_token_id = 248044  # Qwen3.5 <|endoftext|>

    class FakeModel:
        processor = types.SimpleNamespace(tokenizer=FakeTokenizer())

        def generate(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Task"}]
            assert kwargs["eos_token_id"] == 248046
            assert kwargs["pad_token_id"] == 248044
            return types.SimpleNamespace(content="Thought: done\nAction: look")

    backend = object.__new__(QwenTransformersBackend)
    backend.max_request_tokens = 512
    backend.lock = threading.Lock()
    backend.model = FakeModel()
    backend.eos_token_id, backend.pad_token_id = resolve_chat_termination_token_ids(
        backend.model
    )

    output = backend.generate(
        [{"role": "user", "content": "Task"}],
        max_tokens=512,
        temperature=0.0,
        stop=DEFAULT_STOP,
    )
    assert output == "Thought: done\nAction: look"


def test_optional_stubs_cover_only_unused_components():
    install_unused_component_stubs()
    assert "eval_agent.tasks.sciworld" in sys.modules
    assert "eval_agent.envs.sciworld_env" in sys.modules
    assert "eval_agent.envs.webshop_env" in sys.modules
    assert "eval_agent.agents.fastchat_agent" in sys.modules
    assert "eval_agent.tasks.alfworld" not in sys.modules
    assert "eval_agent.envs.alfworld_env" not in sys.modules


def test_downloaded_upstream_is_exact_when_present():
    eto_root = PROJECT_ROOT / "_local" / "upstream" / "ETO"
    if not eto_root.exists():
        pytest.skip("Local upstream ETO checkout is optional")
    report = verify_upstream(eto_root)
    assert report["status"] == "ok", report["errors"]
    assert report["commit"] == EXPECTED_COMMIT
    assert report["protected_files_checked"] == 10


def test_integrity_hash_is_line_ending_independent(tmp_path):
    lf_path = tmp_path / "lf.txt"
    crlf_path = tmp_path / "crlf.txt"
    lf_path.write_bytes(b"first\nsecond\n")
    crlf_path.write_bytes(b"first\r\nsecond\r\n")
    assert sha256_file(lf_path) == sha256_file(crlf_path)


def test_legacy_eto_client_round_trips_through_local_server():
    class FakeBackend:
        def generate(self, messages, *, max_tokens, temperature, stop):
            assert messages == [{"role": "user", "content": "Task"}]
            assert max_tokens == 512
            assert temperature == 0.0
            assert stop == DEFAULT_STOP
            return "Thought: inspect\nAction: go to countertop 1"

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(FakeBackend(), "qwen3.5-0.8b-local"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    previous_openai = sys.modules.get("openai")
    try:
        openai = install_legacy_openai_module()
        openai.api_base = f"http://127.0.0.1:{server.server_port}/v1"
        response = openai.ChatCompletion.create(
            model="qwen3.5-0.8b-local",
            messages=[{"role": "user", "content": "Task"}],
            max_tokens=512,
            temperature=0.0,
            stop=DEFAULT_STOP,
        )
        assert response.choices[0].message["content"].endswith("go to countertop 1")

        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/health", timeout=2
        ) as health_response:
            assert json.loads(health_response.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if previous_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous_openai


def test_runtime_teacher_config_is_ephemeral_and_eto_compatible(tmp_path):
    config_name = write_runtime_model_config(
        tmp_path,
        api_base="https://example.test/v1/",
        api_key="secret-not-for-git",
        model_name="teacher-model",
        max_tokens=512,
        temperature=0.0,
    )
    config = json.loads((tmp_path / f"{config_name}.json").read_text())
    assert config["agent_class"] == "OpenAILMAgent"
    assert config["config"] == {
        "api_base": "https://example.test/v1",
        "api_key": "secret-not-for-git",
        "model_name": "teacher-model",
        "max_tokens": 512,
        "temperature": 0.0,
    }


def test_react_prompt_type_comes_from_alfworld_task_directory():
    game_file = (
        "/data/train/pick_heat_then_place_in_recep-Apple-None-Fridge-1/"
        "trial_123/game.tw-pddl"
    )
    assert prompt_type_for_game(game_file) == "heat"


def test_react_two_shot_uses_official_fixed_example_order():
    eto_root = PROJECT_ROOT / "_local" / "upstream" / "ETO"
    if not eto_root.exists():
        pytest.skip("Local upstream ETO checkout is optional")
    game_file = (
        "/data/train/look_at_obj_in_light-Bowl-None-DeskLamp-1/"
        "trial_123/game.tw-pddl"
    )
    prompt_type, conversations, keys = load_react_two_shot(eto_root, game_file)
    assert prompt_type == "examine"
    assert REACT_EXAMPLE_INDICES == (1, 0)
    assert keys == ["react_examine_1", "react_examine_0"]
    assert len(conversations) == 2
    for conversation in conversations:
        assert conversation[0]["role"] == "user"
        assert conversation[-1]["role"] == "assistant"
        assert "Action: use desklamp" in conversation[-1]["content"]
        assert all(
            message["role"] != "assistant"
            or ("Thought:" in message["content"] and "Action:" in message["content"])
            for message in conversation
        )


def test_react_profile_corrects_only_the_textworld_toggle_command():
    instruction = "5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}"
    assert correct_textworld_action_instruction(instruction) == (
        "5. close {recep}\n6. use {obj}\n7. clean {obj} with {recep}"
    )


def test_rendered_prompt_is_actually_sent_as_system_then_user():
    messages = split_rendered_prompt_into_system_user(
        "rules\n---\nHere are 2 examples.\n\nexamples\n---\n"
        "Now, it's your turn and here is the task.\ncurrent task"
    )
    assert messages == [
        {
            "role": "system",
            "content": "rules\n---\nHere are 2 examples.\n\nexamples",
        },
        {
            "role": "user",
            "content": "Now, it's your turn and here is the task.\ncurrent task",
        },
    ]


def test_normalized_teacher_serialization_round_trips_and_preserves_steps(monkeypatch):
    eto_root = PROJECT_ROOT / "_local" / "upstream" / "ETO"
    raw_path = (
        eto_root
        / "outputs"
        / "qwen3.5-27b"
        / "alfworld_teacher_api_react2"
        / "0.json"
    )
    normalized_path = (
        eto_root
        / "outputs"
        / "qwen3.5-27b"
        / "alfworld_teacher_api_react2_prompt_normalized"
        / "0.json"
    )
    if not raw_path.exists() or not normalized_path.exists():
        pytest.skip("Local teacher trajectories are optional")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    existing_normalized = json.loads(normalized_path.read_text(encoding="utf-8"))

    class State:
        def __init__(
            self,
            reward=None,
            finished=False,
            success=False,
            terminate_reason=None,
        ):
            self.reward = reward
            self.finished = finished
            self.success = success
            self.terminate_reason = terminate_reason
            self.error = None
            self.steps = 0
            self.history = []

        @classmethod
        def load_json(cls, record):
            meta = record["meta"]
            state = cls(
                reward=meta["reward"],
                finished=meta["finished"],
                success=meta["success"],
                terminate_reason=meta["terminate_reason"],
            )
            state.error = meta["error"]
            state.steps = meta["steps"]
            state.history = record["conversations"]
            return state

        def to_dict(self, format="fastchat"):
            assert format == "fastchat"
            return {
                "meta": {
                    "steps": self.steps,
                    "reward": self.reward,
                    "finished": self.finished,
                    "success": self.success,
                    "terminate_reason": self.terminate_reason,
                    "error": self.error,
                },
                "conversations": self.history,
            }

    fake_datatypes = types.ModuleType("eval_agent.utils.datatypes")
    fake_datatypes.State = State
    monkeypatch.setitem(sys.modules, "eval_agent.utils.datatypes", fake_datatypes)

    existing_metadata = existing_normalized["metadata"]
    state = State(
        reward=existing_metadata["reward"],
        finished=existing_metadata["finished"],
        success=existing_metadata["success"],
        terminate_reason=existing_metadata["terminate_reason"],
    )
    state.error = existing_metadata["error"]
    state.steps = existing_metadata["steps"]
    state.history = existing_normalized["messages"]
    report = install_normalized_trajectory_serialization(eto_root)
    assert report["schema_version"] == SCHEMA_VERSION

    normalized = state.to_dict()
    assert normalized["schema_version"] == SCHEMA_VERSION
    assert [message["role"] for message in normalized["messages"][:4]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Here are 2 examples." in normalized["messages"][0]["content"]
    assert normalized["messages"][1]["content"].startswith(
        "Now, it's your turn and here is the task."
    )
    assert sum(
        message["role"] == "assistant" for message in normalized["messages"]
    ) == raw["meta"]["steps"]

    restored = State.load_json(normalized)
    assert restored.steps == state.steps
    assert restored.success == state.success
    assert restored.terminate_reason == state.terminate_reason

    with pytest.raises(ValueError, match="legacy"):
        State.load_json(raw)

    # New teacher runs persist the exact messages sent to the API.
    directly_saved = state.to_dict()
    assert directly_saved["messages"] == state.history
    assert directly_saved["metadata"]["prompt_profile"] == (
        "react-type-2shot-system-user"
    )
