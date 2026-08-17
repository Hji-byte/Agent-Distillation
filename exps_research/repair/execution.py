"""Isolated replay of stored code actions and repaired actions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from smolagents import FinalAnswerTool
from smolagents.local_python_executor import LocalPythonExecutor, fix_final_answer_code
from smolagents.utils import parse_code_blobs


_THOUGHT_RE = re.compile(r"(?m)^[ \t]*Thought:[ \t]*\S")

DEFAULT_REPAIR_IMPORTS = ["numpy", "numpy.linalg", "sympy", "fractions"]


@dataclass
class ExecutedAction:
    code_action: str
    output: Any
    logs: str
    observation: str
    is_final_answer: bool

    def dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = None if self.output is None else str(self.output)
        return data


def parse_action(output_text: str) -> str:
    text = str(output_text or "").strip()
    if not _THOUGHT_RE.search(text):
        raise ValueError("Action is missing a non-empty Thought: line")
    if not text.endswith("</code>"):
        raise ValueError("Action is missing the closing </code> tag")
    return fix_final_answer_code(parse_code_blobs(text, ("<code>", "</code>")))


class IsolatedReplay:
    def __init__(
        self,
        additional_authorized_imports: list[str] | None = None,
        timeout_seconds: int = 30,
    ):
        self.additional_authorized_imports = additional_authorized_imports or list(DEFAULT_REPAIR_IMPORTS)
        self.timeout_seconds = timeout_seconds
        self.executor = self._new_executor()

    def _new_executor(self) -> LocalPythonExecutor:
        executor = LocalPythonExecutor(
            self.additional_authorized_imports,
            timeout_seconds=self.timeout_seconds,
        )
        final_answer = FinalAnswerTool()
        executor.send_tools({final_answer.name: final_answer})
        return executor

    def reset(self) -> None:
        self.executor = self._new_executor()

    def execute_code(self, code_action: str) -> ExecutedAction:
        result = self.executor(fix_final_answer_code(code_action))
        output = None if result.output is None else result.output
        observation = "Execution logs:\n" + result.logs
        observation += "Last output from code snippet:\n" + str(output)
        return ExecutedAction(
            code_action=code_action,
            output=output,
            logs=result.logs,
            observation=observation,
            is_final_answer=result.is_final_answer,
        )

    def replay_prefix(self, trajectory_steps: list[dict[str, Any]], before_step: int) -> list[ExecutedAction]:
        self.reset()
        replayed: list[ExecutedAction] = []
        for index, step in enumerate(trajectory_steps[:before_step]):
            code = step.get("code_action")
            if not code:
                raise ValueError(f"Cannot replay prefix: step {index} has no parsed code_action")
            executed = self.execute_code(str(code))
            if executed.is_final_answer:
                raise ValueError(f"Cannot repair step {before_step}: prefix already finalized at step {index}")
            replayed.append(executed)
        return replayed


__all__ = ["DEFAULT_REPAIR_IMPORTS", "ExecutedAction", "IsolatedReplay", "parse_action"]
