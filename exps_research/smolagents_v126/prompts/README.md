# Prompts

`math_code_agent.yaml` is the smolagents 1.26 CodeAgent prompt used by the
math experiments. It is identical to the official v1.26 prompt except that
the five document/image/retrieval examples were removed; the Python arithmetic
example and every other prompt section remain unchanged.

Pass it to `CodeAgent(prompt_templates=...)`. Do not edit the prompt bundled
with the official smolagents checkout.
