from __future__ import annotations

import sys
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from exps_research.alfworld_eto.integrity import EXPECTED_COMMIT, verify_upstream
from exps_research.alfworld_eto.legacy_openai import install_legacy_openai_module
from exps_research.alfworld_eto.openai_compat_server import make_handler
from exps_research.alfworld_eto.openai_compat_server import DEFAULT_STOP, normalize_stops
from exps_research.alfworld_eto.optional_imports import install_unused_component_stubs
from exps_research.alfworld_eto.run_upstream import write_runtime_model_config
from exps_research.alfworld_eto.react_prompting import (
    REACT_EXAMPLE_INDICES,
    correct_textworld_action_instruction,
    load_react_two_shot,
    prompt_type_for_game,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_normalize_stops_preserves_eto_stop_words():
    assert normalize_stops(None) == DEFAULT_STOP
    assert normalize_stops("END") == ["END"]
    assert normalize_stops(["A", "B"]) == ["A", "B"]
    with pytest.raises(ValueError):
        normalize_stops([1])


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
