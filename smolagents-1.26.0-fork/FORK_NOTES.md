# Fork notes

This directory is based on [huggingface/smolagents](https://github.com/huggingface/smolagents), tag `v1.26.0`, upstream commit `12c1bc820eca50ace6f80a21d90426d41d74f845`.

Project-specific changes include:

- one-step format-correction retry when a CodeAgent response omits the required `Thought:` section;
- retry metadata retained in agent memory and serialized trajectories;
- a text-only local Transformers loading mode for Qwen3.5 models;
- correct placement of Qwen chat-template options such as `enable_thinking`;
- trajectory validation hooks used by the distillation pipeline;
- a non-blocking local Python timeout, with true interruption on Linux/DSW;
- focused regression tests for the modified behavior.

The package name and public API remain `smolagents`, allowing the project adapter to use normal imports. Upstream license and notices are retained. This fork is vendored so experiments resolve a known implementation rather than whichever smolagents version happens to be installed globally.
