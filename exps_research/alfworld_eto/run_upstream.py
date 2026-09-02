from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from exps_research.alfworld_eto.integrity import verify_upstream
from exps_research.alfworld_eto.legacy_openai import install_legacy_openai_module
from exps_research.alfworld_eto.manifest_tasks import install_alfworld_manifest_loader
from exps_research.alfworld_eto.optional_imports import install_unused_component_stubs
from exps_research.alfworld_eto.optional_imports import install_alfworld_environment_exports
from exps_research.alfworld_eto.react_prompting import install_react_two_shot_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ETO_ROOT = PROJECT_ROOT / "_local" / "upstream" / "ETO"
MODEL_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "model"


def assert_server_ready(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Local Qwen server returned HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(
            f"Local Qwen server is not ready at {url}. "
            "Start scripts/alfworld/serve_qwen35_local.sh first."
        ) from exc


def write_runtime_model_config(
    directory: Path,
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Create an ephemeral ETO model config without storing API keys in Git."""
    config_name = "runtime_openai"
    config = {
        "agent_class": "OpenAILMAgent",
        "config": {
            "api_base": api_base.rstrip("/"),
            "api_key": api_key,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    }
    (directory / f"{config_name}.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return config_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original ETO ALFWorld loop with local Qwen")
    parser.add_argument("--eto-root", type=Path, default=DEFAULT_ETO_ROOT)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--part-num", type=int, default=1)
    parser.add_argument("--part-idx", type=int, default=-1)
    parser.add_argument("--exp-name", default="_qwen35_local")
    parser.add_argument("--model-name", default="qwen3.5-0.8b-local")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default="ALFWORLD_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--server-health-url")
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--prompt-profile",
        choices=("upstream", "react-type-2shot"),
        default="upstream",
    )
    args = parser.parse_args()

    report = verify_upstream(args.eto_root, require_data=True)
    if report["status"] != "ok":
        raise RuntimeError("ETO integrity check failed:\n- " + "\n- ".join(report["errors"]))
    health_url = args.server_health_url
    if health_url is None and args.api_base is None:
        health_url = "http://127.0.0.1:8000/health"
    if health_url:
        assert_server_ready(health_url)

    eto_root = args.eto_root.expanduser().resolve()
    sys.path.insert(0, str(eto_root))
    install_legacy_openai_module()
    install_unused_component_stubs()
    install_alfworld_environment_exports()

    # Imported only after compatibility modules are installed. The original
    # ETO main loop, ALFWorld Task/Env, Prompt, and OpenAILMAgent are untouched.
    import eval_agent.main as upstream_main

    if args.prompt_profile == "react-type-2shot":
        prompt_report = install_react_two_shot_prompt(eto_root)
        print("Using ALFWorld prompt profile: " + json.dumps(prompt_report))

    if args.verbose:
        upstream_main.logger.setLevel(logging.INFO)
    elif args.debug:
        upstream_main.logger.setLevel(logging.DEBUG)
    else:
        upstream_main.logger.setLevel(logging.WARNING)

    if args.task_manifest:
        manifest_report = install_alfworld_manifest_loader(
            args.task_manifest, eto_root=eto_root, max_tasks=args.max_tasks
        )
        print("Using ALFWorld task manifest: " + json.dumps(manifest_report))

    with tempfile.TemporaryDirectory(prefix="eto-model-config-") as temp_dir:
        if args.api_base:
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Teacher API key is missing. Set {args.api_key_env} before running."
                )
            agent_path = Path(temp_dir)
            agent_config = write_runtime_model_config(
                agent_path,
                api_base=args.api_base,
                api_key=api_key,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            agent_path = MODEL_CONFIG_DIR
            agent_config = "qwen35_local_openai"

        upstream_args = argparse.Namespace(
            exp_name=args.exp_name,
            exp_path=str(eto_root / "eval_agent" / "configs" / "task"),
            exp_config="alfworld",
            split=args.split,
            part_num=args.part_num,
            part_idx=args.part_idx,
            agent_path=str(agent_path),
            agent_config=agent_config,
            model_name=args.model_name,
            verbose=args.verbose,
            debug=args.debug,
            override=args.override,
            interactive=False,
        )

        previous_cwd = Path.cwd()
        try:
            os.chdir(eto_root)
            upstream_main.main(upstream_args)
        finally:
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
