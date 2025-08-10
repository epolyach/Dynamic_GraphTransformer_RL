"""
Training Package (Cleaned)

This package has been cleaned to remove unused legacy training modules.
All training functionality is now handled inline in run_comparative_study.py.

Legacy training modules have been moved to src/training_backup/.

Note: GAT+RL legacy model comparison uses src_batch.train.train_model
(which forwards to ../GAT_RL/), not the modules that were in this directory.

This directory is reserved for future modular training implementations.
"""

# This package is currently empty - all training is inline in run_comparative_study.py
# Add modular training implementations here as needed

__version__ = "2.0.0"
__all__ = []
