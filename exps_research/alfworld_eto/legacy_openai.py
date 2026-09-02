from __future__ import annotations

import json
import http.client
import sys
import types
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any


class OpenAICompatError(RuntimeError):
    """Base exception exposed through the legacy ``openai.error`` namespace."""


class APIError(OpenAICompatError):
    pass


class Timeout(OpenAICompatError):
    pass


class RateLimitError(OpenAICompatError):
    pass


class ServiceUnavailableError(OpenAICompatError):
    pass


class APIConnectionError(OpenAICompatError):
    pass


def install_legacy_openai_module() -> types.ModuleType:
    """Install the small OpenAI 0.28 surface used by ETO.

    ETO only calls ``openai.ChatCompletion.create``. This local module forwards
    that call to the Qwen server without changing ETO's OpenAILMAgent source.
    """

    module = types.ModuleType("openai")
    module.api_base = "http://127.0.0.1:8000/v1"
    module.api_key = "local"
    module.error = SimpleNamespace(
        APIError=APIError,
        Timeout=Timeout,
        RateLimitError=RateLimitError,
        ServiceUnavailableError=ServiceUnavailableError,
        APIConnectionError=APIConnectionError,
    )

    class ChatCompletion:
        @staticmethod
        def create(**kwargs: Any) -> SimpleNamespace:
            api_base = str(module.api_base).rstrip("/")
            model_name = str(kwargs.get("model", ""))
            if model_name.startswith(("qwen3.5", "qwen3.7")):
                # Match the rest of this project's Qwen agent experiments:
                # ETO needs the visible Thought/Action text in `content`, not
                # native reasoning isolated in `reasoning_content`.
                kwargs.setdefault("enable_thinking", False)
            payload = json.dumps(kwargs).encode("utf-8")
            request = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {module.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw_body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    raise RateLimitError(detail) from exc
                if exc.code >= 500:
                    raise ServiceUnavailableError(detail) from exc
                raise APIError(detail) from exc
            except urllib.error.URLError as exc:
                raise APIConnectionError(str(exc)) from exc
            except TimeoutError as exc:
                raise Timeout(str(exc)) from exc
            except http.client.HTTPException as exc:
                raise APIConnectionError(str(exc)) from exc
            except OSError as exc:
                raise APIConnectionError(str(exc)) from exc

            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise APIError("Model server returned invalid JSON") from exc
            if not isinstance(body, dict):
                raise APIError(
                    f"Model server returned {type(body).__name__}, expected an object"
                )

            choices = []
            raw_choices = body.get("choices", [])
            if not isinstance(raw_choices, list):
                raise APIError("Model server returned a non-list choices field")
            for choice in raw_choices:
                if not isinstance(choice, dict):
                    raise APIError("Model server returned an invalid choice")
                message = choice.get("message", {})
                if not isinstance(message, dict):
                    raise APIError("Model server returned an invalid message")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise APIError(
                        "Model returned no visible content "
                        f"(finish_reason={choice.get('finish_reason')}, "
                        f"message_keys={sorted(message)})"
                    )
                choices.append(SimpleNamespace(message={"content": content}))
            if not choices:
                raise APIError(f"Model server returned no choices: {body}")
            return SimpleNamespace(choices=choices)

    module.ChatCompletion = ChatCompletion
    sys.modules["openai"] = module
    return module
