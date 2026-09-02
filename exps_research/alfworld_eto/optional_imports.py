from __future__ import annotations

import sys
import types


def _install_stub(module_name: str, class_name: str) -> None:
    module = types.ModuleType(module_name)

    class UnusedETOComponent:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{class_name} is unavailable in the ALFWorld-only runtime. "
                "This adapter intentionally installs only ALFWorld dependencies."
            )

    UnusedETOComponent.__name__ = class_name
    setattr(module, class_name, UnusedETOComponent)
    sys.modules[module_name] = module


def install_unused_component_stubs() -> None:
    """Prevent eager imports of ETO components unused by ALFWorld.

    The upstream package imports all three environments at module import time.
    Stubbing the unused ScienceWorld, WebShop, and FastChat modules lets us run
    the original ALFWorld Task, Env, Prompt, Agent, and main loop unchanged.
    """

    _install_stub("eval_agent.tasks.sciworld", "SciWorldTask")
    _install_stub("eval_agent.envs.sciworld_env", "SciWorldEnv")
    _install_stub("eval_agent.envs.webshop_env", "WebShopEnv")
    _install_stub("eval_agent.agents.fastchat_agent", "FastChatAgent")


def install_alfworld_environment_exports() -> None:
    """Restore the public export expected by ETO on newer ALFWorld releases.

    ALFWorld 0.4.x still ships the official ``AlfredTWEnv`` class, but no
    longer re-exports it from ``alfworld.agents.environment``. ETO resolves
    the class from that public package at runtime, so expose the unchanged
    upstream class without editing either project.
    """
    import alfworld.agents.environment as environments

    if not hasattr(environments, "AlfredTWEnv"):
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        environments.AlfredTWEnv = AlfredTWEnv
