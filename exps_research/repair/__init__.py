"""Offline, verifier-grounded local repair for failed math-agent trajectories.

This package is deliberately separate from the baseline teacher-generation and
SFT paths. Importing it has no side effects and does not alter smolagents.
"""

from .localization import RepairCandidate, classify_failure, error_aware_backward_candidates
from .pipeline import RepairConfig, RepairPipeline
from .sft import materialize_accepted_repairs, tokenize_last_assistant_only

__all__ = [
    "RepairCandidate",
    "RepairConfig",
    "RepairPipeline",
    "classify_failure",
    "error_aware_backward_candidates",
    "materialize_accepted_repairs",
    "tokenize_last_assistant_only",
]
