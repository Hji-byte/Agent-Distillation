"""Convert native smolagents 1.26 run records to the project's trajectory schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from smolagents import validate_run_result_for_sft


_MISSING = object()
TRAJECTORY_SCHEMA_VERSION = "smolagents-v126-native-v1"


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _json_safe(value: Any) -> Any:
    """Return JSON-compatible data without depending on smolagents helpers."""
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _json_safe(dict_method())
    return str(value)


def _step_dict(step: Any) -> dict[str, Any]:
    if isinstance(step, Mapping):
        return _json_safe(step)
    dict_method = getattr(step, "dict", None)
    if callable(dict_method):
        return _json_safe(dict_method())
    if is_dataclass(step):
        return _json_safe(asdict(step))
    raise TypeError(f"Unsupported RunResult step type: {type(step).__name__}")


def _text_content(text: Any) -> list[dict[str, str]]:
    return [{"type": "text", "text": "" if text is None else str(text)}]


def _message(role: str, text: Any) -> dict[str, Any]:
    return {"role": role, "content": _text_content(text)}


def _role_name(role: Any) -> str:
    if isinstance(role, Enum):
        role = role.value
    role = str(role)
    if role.startswith("MessageRole."):
        role = role.split(".", 1)[1]
    return role.lower().replace("_", "-")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text", content.get("content", "")))
    if isinstance(content, Sequence):
        return "".join(_content_text(block) for block in content)
    return str(content)


def _infer_system_prompt(steps: list[dict[str, Any]]) -> str | None:
    """Fall back to the first model input when no Agent object is supplied."""
    for step in steps:
        for message in step.get("model_input_messages") or []:
            if _role_name(_read(message, "role", "")) == "system":
                return _content_text(_read(message, "content"))
    return None


def _infer_task(steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        if "task" in step:
            return str(step["task"])
    return None


def _tool_calls(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = step.get("tool_calls") or []
    return [_json_safe(call) for call in calls]


def _tool_call_id(calls: list[dict[str, Any]]) -> str | None:
    if not calls:
        return None
    call_id = calls[0].get("id")
    return str(call_id) if call_id is not None else None


def _error_text(error: Any) -> str:
    if isinstance(error, Mapping):
        message = error.get("message")
        error_type = error.get("type")
        if message and error_type:
            return f"{error_type}: {message}"
        return str(message or error)
    return str(error)


def _structured_action_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep compact 1.26 action metadata for filtering and loss masking."""
    structured_steps: list[dict[str, Any]] = []
    assistant_turn_index = 0
    for step in steps:
        if not any(
            key in step
            for key in ("model_output", "code_action", "observations", "error", "is_final_answer")
        ):
            continue

        model_output = step.get("model_output")
        current_turn = assistant_turn_index if model_output is not None else None
        if model_output is not None:
            assistant_turn_index += 1

        error = step.get("error")
        if error is not None and not isinstance(error, Mapping):
            error = {"type": type(error).__name__, "message": str(error)}

        structured_steps.append(
            {
                "step_number": step.get("step_number"),
                "assistant_turn_index": current_turn,
                "model_output": model_output,
                "code_action": step.get("code_action"),
                "observations": step.get("observations"),
                "error": _json_safe(error),
                "is_final_answer": bool(step.get("is_final_answer", False)),
                "format_retry_count": int(step.get("format_retry_count", 0) or 0),
                "token_usage": _step_token_usage(step.get("token_usage")),
                "duration": _timing_dict(step.get("timing")).get("duration"),
            }
        )
    return structured_steps


def _steps_to_messages(
    steps: list[dict[str, Any]],
    *,
    system_prompt: str,
    task: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [_message("system", system_prompt)]
    original_memory_steps: list[dict[str, Any]] = []
    task_written = False

    for step in steps:
        if "task" in step:
            step_task = str(step.get("task") or task)
            messages.append(_message("user", f"New task:\n{step_task}"))
            task_written = True
            continue

        if "plan" in step:
            plan = str(step.get("plan") or "").strip()
            planning_messages = [
                _message("assistant", plan),
                _message("user", "Now proceed and carry out this plan."),
            ]
            messages.extend(planning_messages)
            original_memory_steps.append({"messages": planning_messages})
            continue

        # FinalAnswerStep contains only ``output``. The answer-producing
        # ActionStep is already represented by model_output/tool response.
        if set(step).issubset({"output"}):
            continue

        if not any(
            key in step
            for key in ("model_output", "code_action", "observations", "error", "tool_calls", "action_output")
        ):
            continue

        model_output = step.get("model_output")
        if model_output is not None:
            messages.append(_message("assistant", str(model_output).strip()))

        calls = _tool_calls(step)
        if calls:
            messages.append(_message("tool-call", "Calling tools:\n" + str(calls)))

        call_id = _tool_call_id(calls)
        call_prefix = f"Call id: {call_id}\n" if call_id else ""

        observations = step.get("observations")
        if observations is not None:
            messages.append(
                _message("tool-response", f"{call_prefix}Observation:\n{observations}")
            )

        error = step.get("error")
        if error is not None:
            retry_instruction = (
                "\nNow let's retry: take care not to repeat previous errors! "
                "If you have retried several times, try a completely different approach.\n"
            )
            messages.append(
                _message(
                    "tool-response",
                    f"{call_prefix}Error:\n{_error_text(error)}{retry_instruction}",
                )
            )

    if not task_written:
        messages.insert(1, _message("user", f"New task:\n{task}"))

    return messages, original_memory_steps


def _timing_dict(timing: Any) -> dict[str, Any]:
    if timing is None:
        return {"start_time": None, "end_time": None, "duration": None}
    data = _json_safe(timing)
    if not isinstance(data, Mapping):
        return {"start_time": None, "end_time": None, "duration": None}
    start = data.get("start_time")
    end = data.get("end_time")
    duration = data.get("duration")
    if duration is None and start is not None and end is not None:
        duration = end - start
    return {"start_time": start, "end_time": end, "duration": duration}


def _token_usage(run_result: Any, steps: list[dict[str, Any]]) -> tuple[int, int]:
    usage = _json_safe(_read(run_result, "token_usage"))
    if isinstance(usage, Mapping):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    input_tokens = 0
    output_tokens = 0
    for step in steps:
        step_usage = step.get("token_usage")
        if isinstance(step_usage, Mapping):
            input_tokens += int(step_usage.get("input_tokens") or 0)
            output_tokens += int(step_usage.get("output_tokens") or 0)
    return input_tokens, output_tokens


def _step_token_usage(usage: Any) -> dict[str, int] | None:
    data = _json_safe(usage)
    if not isinstance(data, Mapping):
        return None
    input_tokens = int(data.get("input_tokens") or 0)
    output_tokens = int(data.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def run_result_to_legacy_log_data(
    run_result: Any,
    *,
    agent: Any | None = None,
    task: str | None = None,
    task_id: str | int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    model_id: str | None = None,
    agent_name: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Convert a smolagents 1.26 ``RunResult`` into the project log format."""
    raw_steps = _read(run_result, "steps", _MISSING)
    if raw_steps is _MISSING:
        raise ValueError("run_result does not contain 'steps'; use return_full_result=True")
    steps = [_step_dict(step) for step in (raw_steps or [])]

    if system_prompt is None and agent is not None:
        memory = _read(agent, "memory")
        prompt_step = _read(memory, "system_prompt")
        system_prompt = _read(prompt_step, "system_prompt")
    system_prompt = system_prompt or _infer_system_prompt(steps)
    if not system_prompt:
        raise ValueError("system prompt is unavailable; pass agent= or system_prompt=")

    task = task or _infer_task(steps)
    if not task:
        raise ValueError("task is unavailable; pass task= or include a TaskStep")

    model = _read(agent, "model") if agent is not None else None
    model_id = model_id or _read(model, "model_id", "unknown")
    agent_name = agent_name or (type(agent).__name__ if agent is not None else "CodeAgent")

    messages, original_memory_steps = _steps_to_messages(
        steps,
        system_prompt=str(system_prompt),
        task=str(task),
    )
    trajectory_steps = _structured_action_steps(steps)

    timing = _timing_dict(_read(run_result, "timing"))
    if timestamp is None:
        start_time = timing.get("start_time")
        timestamp = (
            datetime.fromtimestamp(start_time).strftime("%Y%m%d_%H%M%S")
            if isinstance(start_time, (int, float))
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    input_tokens, output_tokens = _token_usage(run_result, steps)
    state = str(_read(run_result, "state", "unknown"))
    validation = validate_run_result_for_sft(run_result).dict()
    final_answer = _read(run_result, "output")
    step_durations = [
        duration
        for step in steps
        if (duration := _timing_dict(step.get("timing")).get("duration")) is not None
    ]

    metadata: dict[str, Any] = {
        "task": str(task),
        "agent_name": agent_name,
        "model": {"name": type(model).__name__ if model is not None else "unknown", "model_id": model_id},
        "timestamp": timestamp,
        "total_steps": len(steps),
        "success": state == "success",
        "state": state,
        "trajectory_validation": validation,
        "parsing_error_assistant_turns": validation["parsing_error_assistant_turns"],
        "execution_error_assistant_turns": validation["execution_error_assistant_turns"],
        "other_error_assistant_turns": validation["other_error_assistant_turns"],
        "final_answer": None if final_answer is None else str(final_answer),
        "performance": {
            "total_duration": timing.get("duration"),
            "average_step_duration": (
                sum(step_durations) / len(step_durations) if step_durations else 0.0
            ),
        },
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if task_id is not None:
        metadata["task_id"] = task_id

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "source": _json_safe(source_metadata or {}),
        "messages": messages,
        "trajectory_steps": trajectory_steps,
        "original_memory": {
            "system_prompt": {"system_prompt": str(system_prompt)},
            "steps": original_memory_steps,
        },
        "metadata": metadata,
    }


def run_result_to_sft_example(
    run_result: Any,
    *,
    agent: Any | None = None,
    task: str | None = None,
    task_id: str | int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    sft_system_prompt: str | None = None,
    model_id: str | None = None,
    agent_name: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Convert a 1.26 run directly to an SFT example while retaining metadata."""
    log_data = run_result_to_legacy_log_data(
        run_result,
        agent=agent,
        task=task,
        task_id=task_id,
        source_metadata=source_metadata,
        system_prompt=system_prompt,
        model_id=model_id,
        agent_name=agent_name,
        timestamp=timestamp,
    )

    try:
        from exps_research.train_utils.message_utils import prepare_sft_messages
    except ModuleNotFoundError:  # Support running scripts from exps_research/.
        from exps_research.train_utils.message_utils import prepare_sft_messages

    return {
        "schema_version": log_data["schema_version"],
        "source": log_data["source"],
        "messages": prepare_sft_messages(
            log_data["messages"],
            system_prompt=sft_system_prompt,
        ),
        "metadata": log_data["metadata"],
    }
