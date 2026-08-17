"""
Unified experiment execution module
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

from exps_research.unified_framework.processors import (
    get_experiment_processor,
    ExperimentProcessor
)
from exps_research.unified_framework.utils import (
    load_dataset, 
    get_answered_questions, 
    prepare_output_path
)


def load_prompt(prompt_name: str = "teacher_model") -> str:
    """
    Load system prompt from YAML file
    
    Args:
        prompt_name: Name of the prompt file (without extension)
    
    Returns:
        System prompt string
    """
    prompt_file = Path(__file__).resolve().parents[1] / "prompts" / f"{prompt_name}.yaml"
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['system_prompt']


def process_qa_experiment(
    dataset_file: str,
    model_kwargs: Dict[str, Any],
    experiment_type: str = "reasoning",
    output_file: str = None,
    max_workers: int = 4,
    debug: bool = False,
    track_cost: bool = False,
    cost_threshold: float = None,
    fine_tuned: bool = False,
    verbose: bool = False,
    use_process_pool: bool = False,
    use_single_endpoint: bool = False,
    max_samples: int = None,
    **extra_kwargs
) -> Dict:
    """
    Process a QA experiment using the appropriate processor
    
    Args:
        dataset_file: Path to dataset file
        model_kwargs: Model configuration parameters
        experiment_type: Type of experiment ("reasoning" or "agent")
        output_file: Path to output file
        max_workers: Maximum number of concurrent workers
        debug: Whether to run in debug mode
        track_cost: Whether to track cost
        cost_threshold: Cost threshold to stop execution
        fine_tuned: Whether using a fine-tuned model
        verbose: Whether to print verbose output during processing
        use_process_pool: Whether to use ProcessPoolExecutor instead of ThreadPoolExecutor
        use_single_endpoint: Whether to use a single API endpoint for all workers
        **extra_kwargs: Additional experiment-specific parameters
        
    Returns:
        Dictionary with experiment statistics
    """
    # Student benchmarks use one recorded attempt per question. Teacher-data
    # generation keeps the historical behavior of retrying failed runs.
    one_attempt_per_question = bool(
        extra_kwargs.pop("one_attempt_per_question", False)
    )

    # Load dataset
    entries = load_dataset(dataset_file)

    # Check whether question in prefix memory, if prefix memory is enabled
    if prefix_memory := extra_kwargs.get("prefix_memory", None):
        available_questions = list(prefix_memory.keys())
        entries = [entry for entry in entries if entry["question"] in available_questions]
    
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1 when provided")
        entries = entries[:max_samples]

    # Check for already processed questions
    answered_questions = (
        get_answered_questions(
            output_file,
            require_successful_agent_run=(
                experiment_type == "agent" and not one_attempt_per_question
            ),
        )
        if output_file
        else set()
    )
    
    # Filter questions that need to be processed
    entries_todo = [entry for entry in entries if entry["question"] not in answered_questions]
    
    if not entries_todo:
        print("All questions have already been processed. No new questions to generate answers for.")
        return {
            "dataset_file": dataset_file,
            "experiment_type": experiment_type,
            "model_id": model_kwargs["model_id"],
            'total_questions': 0,
            'processed_questions': 0,
            'costs': {
                'total_cost': 0,
                'average_cost_per_question': 0
            },
            'token_usage': {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
            },
        }
    
    print("Output file:", output_file)
    
    # Initialize experiment-specific parameters
    kwargs = {
        'track_cost': track_cost,
        'cost_threshold': cost_threshold,
        'fine_tuned': fine_tuned,
        'verbose': verbose,
        'use_process_pool': use_process_pool,
        'use_single_endpoint': use_single_endpoint,
        **extra_kwargs
    }
    
    # Load system prompt for reasoning experiments
    if experiment_type == "reasoning" and "system_prompt" not in kwargs:
        kwargs["system_prompt"] = load_prompt("teacher_model")
    
    # Get the appropriate processor class and instantiate it
    processor_class = get_experiment_processor(experiment_type)
    processor = processor_class(model_kwargs, **kwargs)
    
    # Process the dataset
    results = processor.process_dataset(
        entries_todo,
        output_file,
        max_workers,
        debug,
        **kwargs  # Don't pass use_local_model directly, it's already in kwargs
    )
    
    # Calculate statistics
    total_questions = len(results)
    total_cost = sum(result.get("cost", 0) for result in results)
    input_tokens = sum(result.get("input_tokens", 0) for result in results)
    output_tokens = sum(result.get("output_tokens", 0) for result in results)
    avg_cost = total_cost / total_questions if total_questions > 0 else 0
    
    stats = {
        "dataset_file": dataset_file,
        "experiment_type": experiment_type,
        "model_id": model_kwargs["model_id"],
        'total_questions': len(entries),
        'processed_questions': total_questions,
        'costs': {
            'total_cost': total_cost,
            'average_cost_per_question': avg_cost
        },
        'token_usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        },
    }
    
    if experiment_type == "agent":
        print(
            "All questions processed. Token usage: "
            f"{input_tokens} input + {output_tokens} output"
        )
    else:
        print(f"All questions processed. Total cost: ${total_cost:.4f}")
    return stats
