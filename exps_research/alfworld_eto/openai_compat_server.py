from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from exps_research.unified_framework.models import setup_model


DEFAULT_STOP = ["\nObservation:", "\nTask:", "\n---"]


def normalize_stops(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_STOP)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("stop must be a string, a list of strings, or null")


def resolve_chat_termination_token_ids(model: Any) -> tuple[Any, Any]:
    """Use the chat tokenizer's terminator instead of the base model config EOS."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        processor = getattr(model, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Local model does not expose a tokenizer")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("Local tokenizer does not define eos_token_id")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
    return eos_token_id, pad_token_id


class QwenTransformersBackend:
    def __init__(
        self,
        model_path: str,
        *,
        lora_path: str | None,
        max_request_tokens: int,
        device_map: str,
        seed: int,
    ) -> None:
        self.max_request_tokens = max_request_tokens
        self.lock = threading.Lock()
        self.model = setup_model(
            model_type="transformers",
            model_id=model_path,
            fine_tuned=lora_path is not None,
            lora_path=lora_path,
            device_map=device_map,
            max_tokens=max_request_tokens,
            temperature=0.0,
            seed=seed,
        )
        self.eos_token_id, self.pad_token_id = resolve_chat_termination_token_ids(self.model)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0 or max_tokens > self.max_request_tokens:
            raise ValueError(f"max_tokens must be between 1 and {self.max_request_tokens}")

        # Qwen3.5's base model config may name <|endoftext|> as EOS while its
        # chat tokenizer closes assistant turns with <|im_end|>.  Passing the
        # tokenizer IDs explicitly prevents generation from continuing into a
        # synthetic next `user` turn after a complete Thought/Action response.
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})

        with self.lock:
            response = self.model.generate(
                messages,
                stop_sequences=stop,
                **generation_kwargs,
            )
        return str(response.content)


def make_handler(backend: QwenTransformersBackend, served_model_name: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Qwen35LocalOpenAICompat/1.0"

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(200, {"status": "ok", "model": served_model_name})
                return
            if self.path == "/v1/models":
                self._write_json(
                    200,
                    {"object": "list", "data": [{"id": served_model_name, "object": "model"}]},
                )
                return
            self._write_json(404, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._write_json(404, {"error": {"message": "Not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                requested_model = request.get("model", served_model_name)
                if requested_model != served_model_name:
                    raise ValueError(
                        f"Unknown model {requested_model!r}; expected {served_model_name!r}"
                    )
                if request.get("stream", False):
                    raise ValueError("Streaming requests are not supported")
                messages = request.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                content = backend.generate(
                    messages,
                    max_tokens=int(request.get("max_tokens", 1024)),
                    temperature=float(request.get("temperature", 0.0)),
                    stop=normalize_stops(request.get("stop")),
                )
                self._write_json(
                    200,
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": served_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._write_json(400, {"error": {"message": str(exc)}})
            except Exception as exc:
                self._write_json(500, {"error": {"message": f"Generation failed: {exc}"}})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local Qwen3.5 for the untouched ETO agent")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="qwen3.5-0.8b-local")
    parser.add_argument("--max-request-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    backend = QwenTransformersBackend(
        args.model_path,
        lora_path=args.lora_path,
        max_request_tokens=args.max_request_tokens,
        device_map=args.device_map,
        seed=args.seed,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(backend, args.served_model_name),
    )
    print(
        f"Qwen server ready at http://{args.host}:{args.port}/v1 "
        f"as {args.served_model_name}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
