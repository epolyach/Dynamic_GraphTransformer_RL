"""
Training library components for CPU-based CVRP model training.

This module contains the core training utilities:
- advanced_trainer: Main training loop with RL techniques
- rollout_baseline: Greedy rollout baseline for REINFORCE
"""

from .advanced_trainer import advanced_train_model
from .rollout_baseline import RolloutBaseline

__all__ = [
    'advanced_train_model',
    'RolloutBaseline',
]
