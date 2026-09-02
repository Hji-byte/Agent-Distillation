from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

# This is the fixed ordering used by the original ReAct ALFWorld notebook.
REACT_EXAMPLE_INDICES = (1, 0)


def prompt_type_for_game(game_file: str) -> str:
    normalized = game_file.replace("\\", "/")
    task_directory = normalized.split("/")[-3]
    for task_prefix, prompt_type in TASK_PREFIXES.items():
        if task_directory.startswith(task_prefix):
            return prompt_type
    raise ValueError(f"Unknown ALFWorld task type in game file: {game_file}")


def correct_textworld_action_instruction(instruction: str) -> str:
    incorrect = "6. toggle {obj} {recep}"
    corrected = "6. use {obj}"
    if corrected in instruction:
        return instruction
    if incorrect not in instruction:
        raise ValueError("Unable to find the ALFWorld toggle action in the instruction")
    return instruction.replace(incorrect, corrected, 1)


def react_demo_to_conversation(demo: str) -> list[dict[str, str]]:
    """Adapt one ReAct transcript to ETO's Thought+Action chat format."""
    event_pattern = re.compile(r"(?m)^(Think|Action|Observation):\s*")
    events = list(event_pattern.finditer(demo))
    if not events:
        raise ValueError("ReAct demonstration contains no trajectory events")

    initial_observation = demo[: events[0].start()].strip()
    if not initial_observation:
        raise ValueError("ReAct demonstration has no initial task observation")

    messages: list[dict[str, str]] = [
        {"role": "user", "content": initial_observation}
    ]
    pending_thoughts: list[str] = []
    last_thought = "Continue with the current plan."

    for index, match in enumerate(events):
        end = events[index + 1].start() if index + 1 < len(events) else len(demo)
        content = demo[match.end() : end].strip()
        event_type = match.group(1)

        if event_type == "Think":
            if content:
                pending_thoughts.append(content)
            continue

        if event_type == "Action":
            if pending_thoughts:
                last_thought = " ".join(pending_thoughts)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Thought: {last_thought}\nAction: {content}",
                }
            )
            pending_thoughts = []
            continue

        # ETO's chat demonstrations end with the successful assistant action;
        # the terminal environment observation is not followed by another turn.
        if index == len(events) - 1:
            continue
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Observation does not follow an action in ReAct demo")
        messages.append({"role": "user", "content": f"Observation: {content}"})

    if messages[-1]["role"] != "assistant":
        raise ValueError("ReAct demonstration must end with its successful action")
    return messages


def load_react_two_shot(
    eto_root: Path, game_file: str
) -> tuple[str, list[list[dict[str, str]]], list[str]]:
    prompt_type = prompt_type_for_game(game_file)
    prompt_path = (
        eto_root
        / "eval_agent"
        / "data"
        / "alfworld"
        / "prompts"
        / "alfworld_3prompts.json"
    )
    prompts: dict[str, str] = json.loads(prompt_path.read_text(encoding="utf-8"))
    keys = [f"react_{prompt_type}_{index}" for index in REACT_EXAMPLE_INDICES]
    conversations = [react_demo_to_conversation(prompts[key]) for key in keys]
    return prompt_type, conversations, keys


def install_react_two_shot_prompt(eto_root: Path) -> dict[str, Any]:
    """Inject task-conditioned ReAct 2-shot prompts without editing ETO source."""
    import eval_agent.envs.alfworld_env as alfworld_env_module

    env_class = alfworld_env_module.AlfWorldEnv
    if getattr(env_class, "_react_two_shot_installed", False):
        return {"status": "already_installed", "profile": "react-type-2shot"}

    original_reset = env_class.reset
    original_prompt_with_icl = alfworld_env_module.prompt_with_icl
    resolved_eto_root = eto_root.expanduser().resolve()

    def prompt_with_two_examples(instruction, raw_icl, cur_task, icl_num=2):
        return original_prompt_with_icl(instruction, raw_icl, cur_task, icl_num=2)

    def reset_with_react_two_shot(self):
        prompt_type, conversations, keys = load_react_two_shot(
            resolved_eto_root, self.task.game_file
        )
        self.instruction = correct_textworld_action_instruction(self.instruction)
        self.raw_icl = conversations
        self.react_prompt_type = prompt_type
        self.react_prompt_keys = keys
        return original_reset(self)

    alfworld_env_module.prompt_with_icl = prompt_with_two_examples
    env_class.reset = reset_with_react_two_shot
    env_class._react_two_shot_installed = True
    return {
        "status": "installed",
        "profile": "react-type-2shot",
        "example_indices": list(REACT_EXAMPLE_INDICES),
        "action_instruction": "use {obj}",
    }
