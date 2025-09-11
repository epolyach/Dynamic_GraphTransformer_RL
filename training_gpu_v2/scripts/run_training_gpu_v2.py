#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training_gpu_v2.lib.trainer_v2 import train_v2
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run v2 GPU trainer (simple, GPU-only)")
    parser.add_argument('--config', type=str, default='configs/tiny_1.yaml', help='Config YAML path')
    parser.add_argument('--model', type=str, default='GT+RL', help='Model name (GAT+RL, GT+RL, DGT+RL)')
    parser.add_argument('--epochs', type=int, default=10, help='Override epochs to run')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory root for results')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for first N epochs')
    parser.add_argument('--profile_epochs', type=int, default=10, help='Number of epochs to profile when --profile is set')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. v2 trainer requires an NVIDIA GPU.')

    profile_epochs: Optional[int] = args.profile_epochs if args.profile else None

    out = train_v2(
        config_path=args.config,
        model_name=args.model,
        out_dir=args.output_dir,
        epochs_override=args.epochs,
        profile_epochs=profile_epochs,
    )

    print(f"v2 training finished. CSV: {out.csv_path}")


if __name__ == '__main__':
    main()

