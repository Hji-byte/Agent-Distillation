"""Fail fast on cloud configuration mistakes before loading the student model."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REQUIRED_MODULES = ("torch", "transformers", "peft", "smolagents", "openai", "dotenv")


def validate_repair_inputs(
    *,
    adapter_path: str | Path,
    model_path: str | Path,
    dataset_path: str | Path,
    max_samples: int,
    require_cuda: bool = True,
    check_imports: bool = True,
) -> dict:
    adapter = Path(adapter_path).resolve()
    model = Path(model_path).resolve()
    dataset = Path(dataset_path).resolve()
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    adapter_config = adapter / "adapter_config.json"
    adapter_weights = next(
        (path for path in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin") if path.is_file()),
        None,
    )
    model_config = model / "config.json"
    for path, description in (
        (adapter_config, "LoRA adapter config"),
        (adapter_weights, "LoRA adapter weights"),
        (model_config, "base-model config"),
        (dataset, "repair dataset"),
    ):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")

    json.loads(adapter_config.read_text(encoding="utf-8"))
    json.loads(model_config.read_text(encoding="utf-8"))
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Repair dataset must contain an examples list")
    if max_samples > len(examples):
        raise ValueError(f"Requested {max_samples} samples but dataset contains only {len(examples)}")
    questions = [str(example.get("question", "")) for example in examples[:max_samples]]
    if any(not question for question in questions):
        raise ValueError("Every selected repair example must have a non-empty question")
    if len(set(questions)) != len(questions):
        raise ValueError("Selected repair examples contain duplicate questions")

    if check_imports:
        for module_name in REQUIRED_MODULES:
            importlib.import_module(module_name)
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is unavailable; local S0 evaluation and continuation require a GPU")

    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "dataset_examples": len(examples),
        "selected_examples": max_samples,
        "adapter_weights": str(adapter_weights),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_samples", type=int, required=True)
    args = parser.parse_args()
    result = validate_repair_inputs(
        adapter_path=args.adapter,
        model_path=args.model,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
