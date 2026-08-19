"""
Agent experiment processor for tool-based experiments
"""

import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .base import ExperimentProcessor
from exps_research.smolagents_v126 import run_result_to_legacy_log_data
from exps_research.unified_framework.models import setup_model
from smolagents import CodeAgent, __version__ as smolagents_version


MATH_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "smolagents_v126"
    / "prompts"
    / "math_code_agent.yaml"
)
MATH_PROMPT_SHA256 = hashlib.sha256(MATH_PROMPT_PATH.read_bytes()).hexdigest()
HARD_TRUNCATION_REASONS = {"parsing_error", "incomplete_action_format"}


def _is_hard_truncated(log_data: dict[str, Any], max_tokens: int | None) -> bool:
    """Return whether a malformed action hit the configured output ceiling."""
    if not max_tokens:
        return False
    validation = log_data.get("metadata", {}).get("trajectory_validation", {})
    reasons = set(validation.get("reasons", []))
    if not reasons.intersection(HARD_TRUNCATION_REASONS):
        return False
    return any(
        (step.get("token_usage") or {}).get("output_tokens", 0) >= max_tokens
        for step in log_data.get("trajectory_steps", [])
    )


def _attempt_summary(
    log_data: dict[str, Any],
    *,
    max_tokens: int | None,
    selected: bool,
) -> dict[str, Any]:
    metadata = log_data.get("metadata", {})
    validation = metadata.get("trajectory_validation", {})
    return {
        "max_tokens": max_tokens,
        "state": metadata.get("state"),
        "valid_structure": bool(validation.get("valid", False)),
        "validation_reasons": list(validation.get("reasons", [])),
        "hard_truncated": _is_hard_truncated(log_data, max_tokens),
        "token_usage": dict(metadata.get("token_usage", {})),
        "selected": selected,
    }


class AgentExperimentProcessor(ExperimentProcessor):
    """
    Processor for agent experiments

    This processor handles agent-based experiments where the model
    interacts with tools to solve problems.
    """

    def __init__(self, model_kwargs: Dict[str, Any], **kwargs):
        """Initialize the agent experiment processor with a rich console if available"""
        super().__init__(model_kwargs, **kwargs)
        if RICH_AVAILABLE:
            self.console = Console()

    def process_entry(self, entry: Dict, model, **kwargs) -> Dict:
        """
        Process an agent experiment entry

        Args:
            entry: Dictionary containing a question
            model: Model instance
            **kwargs: Additional parameters including:
                - max_steps: Maximum number of steps for the agent
                - fine_tuned: Whether using a fine-tuned model
                - set_timeout: Whether to set timeouts for code execution
                - verbose_worker: Whether this worker should display verbose output

        Returns:
            Processed result dictionary
        """
        if self.cost_tracker.stop_requested:
            return None

        # Get experiment parameters
        max_steps = kwargs.get('max_steps', 5)
        retry_max_tokens = kwargs.get('retry_max_tokens')
        use_planning = kwargs.get('use_planning', False)
        prefix_memory = kwargs.get('prefix_memory', None)
        cot_memory = kwargs.get('cot_memory', None)
        # Determine if this worker should show verbose output
        verbose_worker = kwargs.get('verbose_worker', True)
        should_show_output = self.verbose and verbose_worker

        # Set verbosity level for the agent based on whether this is the designated verbose worker
        verbosity_level = 2 if should_show_output else 0

        if should_show_output and RICH_AVAILABLE:
            self.console.rule(f"[bold blue]Processing Agent Question")
            self.console.print(Panel(entry['question'], title="Question", border_style="green"))

        # This project focuses exclusively on mathematical code execution.
        # No retrieval or web-search tools are registered with the agent.
        tools = []

        with MATH_PROMPT_PATH.open(encoding="utf-8") as prompt_file:
            prompt_templates = yaml.safe_load(prompt_file)

        # Use the native smolagents 1.26 <code>...</code> protocol.
        agent_kwargs = {
            "max_steps": max_steps,
        }
        if use_planning:
            agent_kwargs = {
                "planning_interval": 10,
                "max_steps": max_steps + 1
            }
        question = entry["question"]
        instruction = "\n\nFor math problems that are not multiple-choice, always output the final answer using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), not a decimal approximation (e.g., \\boxed{0.642857})."

        try:
            # Run agent with appropriate prompting
            _question = question + instruction

            if should_show_output and RICH_AVAILABLE:
                self.console.print("[bold cyan]Running agent with max_steps:[/bold cyan]", str(max_steps))

            if cot_memory:
                existing_cot = cot_memory.get(question, None)
                if existing_cot:
                    cot_guide = "<reference>\nUse this REFERENCE solution for solving problem. DO NOT directly mention the reference solution in your solution:\n\n" + existing_cot + "</reference>"
                    _question = question + cot_guide + instruction

            source_metadata = {
                key: entry[key]
                for key in ("id", "dataset_name", "split", "level", "type")
                if key in entry
            }

            def run_once(run_model):
                run_agent = CodeAgent(
                    tools=tools,
                    model=run_model,
                    prompt_templates=prompt_templates,
                    additional_authorized_imports=["numpy", "numpy.linalg", "sympy", "fractions"],
                    required_action_prefix="Thought:",
                    max_action_format_retries=1,
                    verbosity_level=verbosity_level,
                    **agent_kwargs,
                )
                if prefix_memory:
                    run_agent.register_prefix([prefix_memory.get(question)])
                run_result = run_agent.run(_question, return_full_result=True)
                run_log_data = run_result_to_legacy_log_data(
                    run_result,
                    agent=run_agent,
                    task=_question,
                    task_id=entry.get("id"),
                    source_metadata=source_metadata,
                )
                return run_agent, run_result.output, run_log_data

            primary_max_tokens = self.model_kwargs.get("max_tokens")
            agent, result, log_data = run_once(model)
            attempts = [
                _attempt_summary(
                    log_data,
                    max_tokens=primary_max_tokens,
                    selected=True,
                )
            ]
            used_retry = False

            if retry_max_tokens and _is_hard_truncated(log_data, primary_max_tokens):
                if should_show_output:
                    message = (
                        f"Hard truncation detected at {primary_max_tokens} tokens; "
                        f"retrying once with {retry_max_tokens}."
                    )
                    if RICH_AVAILABLE:
                        self.console.print(f"[yellow]{message}[/yellow]")
                    else:
                        print(message)

                if self.model_kwargs.get("model_type") == "transformers":
                    # Reuse the already-loaded local checkpoint. Loading a
                    # second TransformersModel for one retry duplicates GPU
                    # memory and can OOM on smaller cards.
                    missing = object()
                    previous_limit = model.kwargs.get("max_new_tokens", missing)
                    model.kwargs["max_new_tokens"] = retry_max_tokens
                    try:
                        retry_agent, retry_result, retry_log_data = run_once(model)
                    finally:
                        if previous_limit is missing:
                            model.kwargs.pop("max_new_tokens", None)
                        else:
                            model.kwargs["max_new_tokens"] = previous_limit
                    retry_model = model
                else:
                    retry_model_kwargs = dict(self.model_kwargs)
                    retry_model_kwargs["max_tokens"] = retry_max_tokens
                    retry_model = setup_model(**retry_model_kwargs)
                    retry_agent, retry_result, retry_log_data = run_once(retry_model)
                attempts[0]["selected"] = False
                attempts.append(
                    _attempt_summary(
                        retry_log_data,
                        max_tokens=retry_max_tokens,
                        selected=True,
                    )
                )
                agent, result, log_data = retry_agent, retry_result, retry_log_data
                model = retry_model
                used_retry = True

            token_usage = log_data["metadata"]["token_usage"]
            log_data["metadata"]["generation_config"] = {
                "smolagents_version": smolagents_version,
                "code_protocol": "native_code_tags",
                "required_action_prefix": "Thought:",
                "max_action_format_retries": 1,
                "prompt": str(MATH_PROMPT_PATH.relative_to(Path(__file__).resolve().parents[3])),
                "prompt_sha256": MATH_PROMPT_SHA256,
                "model_id": model.model_id,
                "temperature": self.model_kwargs.get("temperature"),
                "seed": self.model_kwargs.get("seed"),
                "max_tokens": retry_max_tokens if used_retry else primary_max_tokens,
                "initial_max_tokens": primary_max_tokens,
                "retry_max_tokens": retry_max_tokens,
                "max_steps": max_steps,
            }
            log_data["metadata"]["used_truncation_retry"] = used_retry
            log_data["metadata"]["generation_attempts"] = attempts

            consumed_input_tokens = sum(
                attempt.get("token_usage", {}).get("input_tokens", 0)
                for attempt in attempts
            )
            consumed_output_tokens = sum(
                attempt.get("token_usage", {}).get("output_tokens", 0)
                for attempt in attempts
            )

            # Clean up memory to make logs more compact
            for step in agent.memory.steps:
                if hasattr(step, 'agent_memory'):
                    step.agent_memory = None

            # Display final answer for verbose worker
            if should_show_output and RICH_AVAILABLE:
                try:
                    self.console.print(Panel(result, title="[bold green]Agent Final Answer[/bold green]", border_style="green"))
                    self.console.print(f"[bold]Total steps:[/bold] {len(agent.memory.steps)}")
                    self.console.print(
                        "[bold]Token usage:[/bold] "
                        f"{token_usage['input_tokens']} input + "
                        f"{token_usage['output_tokens']} output"
                    )
                except:
                    print(result)

            # Create result dictionary
            annotated_result = {
                "model_id": model.model_id,
                "question": question,
                "generated_answer": result,
                "true_answer": entry.get("answer", None),
                "log_data": log_data,
                "input_tokens": consumed_input_tokens,
                "output_tokens": consumed_output_tokens,
                "selected_input_tokens": token_usage["input_tokens"],
                "selected_output_tokens": token_usage["output_tokens"],
            }
        except Exception as e:
            error_traceback = traceback.format_exc()
            # Only print errors for the verbose worker
            if should_show_output:
                if RICH_AVAILABLE:
                    self.console.print(f"[bold red]Error processing question:[/bold red] {question}")
                    self.console.print(f"[red]{str(e)}[/red]")
                else:
                    print(f"Error processing question: {question}", str(e))

            annotated_result = {
                "model_id": model.model_id,
                "question": question,
                "generated_answer": None,
                "true_answer": entry.get("answer", None),
                "error": str(e),
                "error_type": type(e).__name__,
                "error_traceback": error_traceback,
                "log_data": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        return annotated_result
