# Unified Experiment Framework

This module provides a unified framework for running both reasoning-based and agent-based experiments on question-answering datasets.

## Structure

- `models.py`: Common model setup code for different model types (OpenAI, VLLM)
- `utils.py`: Shared utility functions for cost calculation, file handling, etc.
- `experiment.py`: Core experiment execution logic with support for both reasoning and agent-based experiments
- `run_reasoning.py`: Main script for running reasoning experiments
- `run_agent.py`: Main script for running agent experiments

## Usage

### Reasoning Experiments

```bash
python -m exps_research.unified_framework.run_reasoning \
  --data_path data_processor/math_dataset/train/math_1000_20250414.json \
  --model_type openai \
  --model_id qwen3.7-plus \
  --parallel_workers 4 \
  --multithreading \
  --temperature 0.0 \
  --seed 42 \
  --track_cost
```

### Math Code-Agent Experiments

Set `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL` in the environment, then run:

```powershell
python -m exps_research.unified_framework.run_experiment `
  --experiment_type agent `
  --data_path data_processor/math_dataset/train/math_1000_20250414.json `
  --model_type openai `
  --model_id qwen3.7-plus `
  --max_steps 5 `
  --max_samples 1 `
  --temperature 0.0 `
  --seed 42 `
  --do_filtering
```

The agent has no retrieval or web-search tools. It uses
`exps_research/smolagents_v126/prompts/math_code_agent.yaml`, which retains the
native smolagents 1.26 CodeAgent protocol and one math example while removing
unrelated retrieval, browsing, document, and image examples. It can only
execute Python code with the imports authorized in `processors/agent.py`.

## Common Parameters

Both experiment types support the following common parameters:

- `--data_path`: Path to the dataset file
- `--model_type`: Type of model to use (`openai`, `transformers`, or `vllm`)
- `--model_id`: Model ID to use (e.g., `gpt-4o-mini`, `gpt-4o`)
- `--parallel_workers`: Maximum number of concurrent threads to use
- `--multithreading`: Run in multithreading mode
- `--temperature`: Sampling temperature (default: 0.0)
- `--seed`: Random seed (default: 42)
- `--debug`: Run in debug mode with limited questions
- Agent experiments record input/output token counts. Monetary prices are not
  hard-coded because API pricing can change.
- `--fine_tuned`: Whether using a fine-tuned model
- `--lora_folder`: The folder for trained LoRA weights and logs
- `--log_folder`: Base folder for storing results
- `--max_samples`: Only process the first N samples

### Local LoRA Evaluation

After QLoRA training, evaluate the adapter with the same MATH500 agent setup
used by the untouched baseline:

```powershell
.\scripts\inference\run_local_qwen35_finetuned.ps1 `
  -AdapterPath ".\training_outputs\qwen3.5-0.8B\agent_baseline_2epochs_qlora"
```

The base checkpoint remains unchanged. Pass its location through `-ModelPath`
or `AGENT_DISTILLATION_MODEL_PATH`; the script attaches the saved PEFT adapter
and writes the evaluation results below the adapter directory.

## Experiment-specific Parameters

### Agent Experiments
- `--max_steps`: Maximum number of steps for the agent (default: 5)

## Extending the Framework

To add support for new experiment types or models:

1. Add any new model types to `models.py`
2. Create new processing functions in `experiment.py` 
3. Update the common utilities in `utils.py` as needed
4. Create a new run script for your experiment type

## Benefits of the Unified Framework

- **Code Reuse**: Common functionality shared between experiment types
- **Consistent Interface**: Similar parameters and output formats
- **Maintainability**: Easier to maintain and extend with modular design
- **Parallel Execution**: Built-in support for multithreading
- **Token Tracking**: Input/output token totals for Agent generation
