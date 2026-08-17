"""
Unified model setup for experiments
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

from dotenv import load_dotenv
from smolagents import OpenAIModel, TransformersModel, VLLMModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def setup_model(
    model_type: str = "openai", 
    model_id: str = None, 
    fine_tuned: bool = False,
    local_device_id: int = -1,
    lora_path: str = None,
    **kwargs
) -> Union[OpenAIModel, TransformersModel, VLLMModel]:
    """
    Initialize a model for experiments
    
    Args:
        model_type: Type of model to use ("openai" or "vllm")
        model_id: Model ID to use (e.g., gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct)
        fine_tuned: Whether to use a fine-tuned model
        **kwargs: Additional keyword arguments for model initialization
    
    Returns:
        Initialized model
    """
    default_models = {
        "openai": "gpt-4o-mini",
        "transformers": "Qwen/Qwen3.5-0.8B",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }
    model_id = model_id or default_models.get(model_type)    
    if model_type == "openai":
        api_base = os.getenv("DASHSCOPE_BASE_URL")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key or not api_base:
            raise RuntimeError(
                "Missing DASHSCOPE_API_KEY or DASHSCOPE_BASE_URL. "
                f"Create {PROJECT_ROOT / '.env'} from .env.example."
            )

        # Keep the agent's visible Thought/Code protocol in `content` instead
        # of placing Qwen3.5/Qwen3.7 reasoning in a separate
        # ``reasoning_content`` field.
        if model_id.startswith(("qwen3.5", "qwen3.7")):
            kwargs.setdefault("extra_body", {"enable_thinking": False})

        return OpenAIModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            **kwargs
        )
    elif model_type == "transformers":
        if fine_tuned and not lora_path:
            raise ValueError("lora_path is required for fine-tuned Transformers evaluation.")
        # The unified runner also serves OpenAI/vLLM backends, whose request
        # arguments are not all valid ``transformers.generate`` arguments.
        # Normalize them before constructing the local model.
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        kwargs.pop("n", None)  # Local baseline evaluation generates one answer.
        seed = kwargs.pop("seed", None)
        if seed is not None:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if kwargs.get("temperature") == 0.0:
            # Greedy decoding is deterministic; sampling-only parameters are
            # invalid or ignored when do_sample=False.
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs["do_sample"] = False
        apply_chat_template_kwargs = kwargs.pop("apply_chat_template_kwargs", {})
        apply_chat_template_kwargs.setdefault("enable_thinking", False)
        model = TransformersModel(
            model_id=model_id,
            device_map=kwargs.pop("device_map", "cuda"),
            torch_dtype=kwargs.pop("torch_dtype", "auto"),
            trust_remote_code=kwargs.pop("trust_remote_code", True),
            text_only=True,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            **kwargs,
        )
        if fine_tuned:
            from peft import PeftModel

            adapter_path = Path(lora_path).expanduser().resolve()
            if not adapter_path.exists():
                raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
            model.model = PeftModel.from_pretrained(model.model, adapter_path)
            model.model.eval()
        return model
    elif model_type == "vllm":
        if int(local_device_id) >= 0:
            if fine_tuned:
                raise ValueError(
                    "smolagents 1.26 VLLMModel does not load a PEFT adapter directly; "
                    "serve the adapter through a vLLM OpenAI-compatible endpoint instead."
                )
            return VLLMModel(model_id=model_id, **kwargs)

        # smolagents 1.26 removed the paper fork's VLLMServerModel. A vLLM
        # server is an OpenAI-compatible endpoint and uses OpenAIModel.
        api_base = kwargs.pop("api_base", "http://127.0.0.1:8000/v1")
        api_key = kwargs.pop("api_key", "token-abc")
        served_model_id = "finetune" if fine_tuned else model_id
        return OpenAIModel(
            model_id=served_model_id,
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
