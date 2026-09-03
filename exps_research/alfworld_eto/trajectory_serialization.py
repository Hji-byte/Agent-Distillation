"""Strict runtime serialization for normalized ALFWorld trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exps_research.alfworld_eto.react_prompting import (
    load_react_two_shot,
    prompt_type_for_game,
)


SCHEMA_VERSION = "alfworld-eto-react2-system-user-v1"


def install_normalized_trajectory_serialization(eto_root: Path) -> dict[str, Any]:
    """Persist and resume only the normalized system/user runtime format."""
    from eval_agent.utils.datatypes import State

    if getattr(State, "_normalized_teacher_serialization_installed", False):
        return {"status": "already_installed", "schema_version": SCHEMA_VERSION}

    resolved_eto_root = eto_root.expanduser().resolve()

    def to_dict(self, format: str = "fastchat"):
        if format != "fastchat":
            raise ValueError("Normalized ALFWorld trajectories use fastchat persistence only")
        if not self.history or self.history[0].get("role") != "system":
            raise ValueError(
                "Refusing to save a non-normalized ALFWorld trajectory; "
                "the first runtime message must be system"
            )
        meta = {
            "steps": self.steps,
            "reward": self.reward,
            "finished": self.finished,
            "success": self.success,
            "terminate_reason": self.terminate_reason,
            "error": self.error,
        }
        prompt_type = prompt_type_for_game(self.error)
        _, _, demo_keys = load_react_two_shot(resolved_eto_root, self.error)
        return {
            "schema_version": SCHEMA_VERSION,
            "messages": self.history,
            "supervision": "all_assistant_turns",
            "metadata": {
                **meta,
                "prompt_profile": "react-type-2shot-system-user",
                "prompt_type": prompt_type,
                "react_example_keys": demo_keys,
            },
        }

    @classmethod
    def load_json(cls, json_dict: dict[str, Any]):
        if json_dict.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Refusing to resume a legacy ALFWorld trajectory in a normalized run"
            )

        metadata = json_dict.get("metadata")
        messages = json_dict.get("messages")
        if not isinstance(metadata, dict) or not isinstance(messages, list):
            raise ValueError("Normalized trajectory must contain metadata and messages")
        state = cls(
            reward=metadata.get("reward"),
            finished=bool(metadata.get("finished", False)),
            success=bool(metadata.get("success", False)),
            terminate_reason=metadata.get("terminate_reason"),
        )
        state.error = metadata.get("error")
        state.steps = int(metadata.get("steps", 0))
        state.history = messages
        return state

    State.to_dict = to_dict
    State.load_json = load_json
    State._normalized_teacher_serialization_installed = True
    return {"status": "installed", "schema_version": SCHEMA_VERSION}


__all__ = ["SCHEMA_VERSION", "install_normalized_trajectory_serialization"]
