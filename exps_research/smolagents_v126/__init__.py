"""Compatibility helpers for running the experiments with smolagents 1.26.x."""

from .trajectory_adapter import run_result_to_legacy_log_data, run_result_to_sft_example

__all__ = [
    "run_result_to_legacy_log_data",
    "run_result_to_sft_example",
]
