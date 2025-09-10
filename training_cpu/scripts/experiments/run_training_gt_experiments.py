#!/usr/bin/env python3
"""
GT Experiment Driver (single-run or small sweeps)

Purpose
- Provide a focused entrypoint for GT+RL experiments
- Allow lightweight hyperparameter sweeps via CLI (heads, layers, hidden, dropout, ff-mult)
- Toggle optimizer experiment features (Adam β2 scaling, advantage normalization)
- Preserve the standard training pipeline and artifacts

Notes
- Uses the experimental trainer (advanced_trainer_opt_experiments) to keep optimizer tweaks isolated
- Defaults to configs/tiny_opt.yaml to match prior optimizer experiment work

Examples
Single-run (override a few params):
  python training_cpu/scripts/run_training_gt_experiments.py \
    --config ../../configs/tiny_opt.yaml \
    --hidden 256 --heads 4 --layers 4 --dropout 0.1 --ff-mult 4 \
    --batch-size 32 --lr 1e-4 --epochs 60 \
    --override-beta2 true --normalize-advantages batch

Small sweep over heads and layers:
  python training_cpu/scripts/run_training_gt_experiments.py \
    --config ../../configs/tiny_opt.yaml \
    --name arch_sweep --heads 2,4 --layers 3,4 --hidden 128 \
    --batch-size 32 --epochs 50 --dry-run     # preview

  python training_cpu/scripts/run_training_gt_experiments.py \
    --config ../../configs/tiny_opt.yaml \
    --name arch_sweep --heads 2,4 --layers 3,4 --hidden 128 \
    --batch-size 32 --epochs 50               # run
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

# Add project root to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
# Make training_cpu/lib importable
training_cpu_path = Path(__file__).parent.parent
sys.path.insert(0, str(training_cpu_path))
from lib.advanced_trainer_opt_experiments import advanced_train_model_opt
from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.utils.seeding import set_seeds


def model_key(name: str) -> str:
    return name.lower().replace('+', '_').replace('-', '_').replace(' ', '_')


def setup_logging(config=None):
    import logging
    level = logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', format_str)
        level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)


class IncrementalCSVWriter:
    def __init__(self, run_dir: str, model_name: str, config: dict, logger):
        self.csv_dir = os.path.join(run_dir, 'csv')
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_path = os.path.join(self.csv_dir, f'history_{model_key(model_name)}.csv')
        self.val_freq = int(config['training']['validation_frequency'])
        self.logger = logger
        pd.DataFrame(columns=[
            'epoch','train_loss','train_cost','val_cost','learning_rate','temperature','baseline_type','baseline_value'
        ]).to_csv(self.csv_path, index=False)
        self.baseline_type = config.get('baseline', {}).get('type', 'mean')
        self.logger.info(f'[GT-EXP] Created CSV history file: {self.csv_path}')
    
    def write_epoch(self, epoch: int, train_loss: float, train_cost: float, val_cost: Optional[float], learning_rate: float, temperature: float, baseline_value: float = None):
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_cost': train_cost,
            'val_cost': val_cost if val_cost is not None else float('nan'),
            'learning_rate': learning_rate,
            'temperature': temperature,
            'baseline_type': self.baseline_type,
            'baseline_value': baseline_value if baseline_value is not None else float('nan')
        }
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)


def parse_list(arg: Optional[str], cast):
    if arg is None:
        return None
    vals = [x.strip() for x in str(arg).split(',') if x.strip() != '']
    return [cast(v) for v in vals]


def format_float(x: float) -> str:
    # Compact float for folder names
    if x == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(x))))
    if -3 <= exp <= 3:
        return f'{x:g}'
    return f'{x:.1e}'.replace('+0', '+').replace('-0', '-')


def combo_label(h: Optional[int], l: Optional[int], d: Optional[int], dr: Optional[float], ff: Optional[int], bs: Optional[int], lr: Optional[float], e: Optional[int], b2: Optional[bool], norm: Optional[str]) -> str:
    parts = []
    if h is not None: parts.append(f'h{h}')
    if l is not None: parts.append(f'l{l}')
    if d is not None: parts.append(f'd{d}')
    if dr is not None: parts.append(f'dr{format_float(dr)}')
    if ff is not None: parts.append(f'ff{ff}')
    if bs is not None: parts.append(f'bs{bs}')
    if lr is not None: parts.append(f'lr{format_float(lr)}')
    if e is not None: parts.append(f'e{e}')
    if b2 is not None: parts.append('b2Y' if b2 else 'b2N')
    if norm is not None: parts.append(f'norm{norm}')
    return '_'.join(parts) if parts else 'default'


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deep_update(cfg, overrides) if overrides else cfg
    return cfg


def ensure_sections(cfg: Dict[str, Any]):
    cfg.setdefault('model', {})
    cfg.setdefault('training', {})
    cfg.setdefault('inference', {})
    cfg.setdefault('baseline', {})
    cfg.setdefault('experiment', {})
    cfg.setdefault('training_advanced', {})
    cfg.setdefault('optimizer_experiments', {})


def build_run_dir(base_dir: Path, name: Optional[str], label: str) -> Path:
    if name:
        return base_dir / 'experiments' / name / label
    return base_dir / 'experiments' / label


def save_model(run_dir: str, model_name: str, model: torch.nn.Module, results: Dict[str, Any], training_time: float, config: dict, logger) -> None:
    pytorch_dir = os.path.join(run_dir, 'pytorch')
    os.makedirs(pytorch_dir, exist_ok=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    payload = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'config': config,
        'history': results.get('history', {}),
        'artifacts': results.get('artifacts', {}),
        'training_time': training_time,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': str(model.__class__.__name__),
    }
    model_path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
    torch.save(payload, model_path)
    logger.info(f'[GT-EXP] Saved model: {model_path}')


def parse_args():
    p = argparse.ArgumentParser(description='GT+RL experiment driver (single-run or small sweeps)')
    p.add_argument('--config', type=str, default='../../configs/tiny_opt.yaml', help='Base config path (deep-merged with defaults)')
    p.add_argument('--name', type=str, default=None, help='Experiment group name (for folder structure)')
    p.add_argument('--output-dir', type=str, default=None, help='Override output directory from config')
    p.add_argument('--force-retrain', action='store_true', help='Retrain even if a run directory already exists')
    p.add_argument('--dry-run', action='store_true', help='Only print planned runs without training')

    # Architecture overrides (single value or comma-separated lists for sweeps)
    p.add_argument('--heads', type=str, default=None, help='Transformer heads e.g. "4" or "2,4,8"')
    p.add_argument('--layers', type=str, default=None, help='Transformer layers e.g. "3" or "3,4"')
    p.add_argument('--hidden', type=str, default=None, help='Hidden dim e.g. "128" or "128,256"')
    p.add_argument('--dropout', type=str, default=None, help='Transformer dropout e.g. "0.1" or "0.0,0.1"')
    p.add_argument('--ff-mult', type=str, default=None, help='Feedforward multiplier e.g. "4"')

    # Training overrides
    p.add_argument('--batch-size', type=str, default=None, help='Batch size e.g. "32" or "1,8,32"')
    p.add_argument('--epochs', type=str, default=None, help='Num epochs e.g. "50"')
    p.add_argument('--lr', type=str, default=None, help='Learning rate e.g. "1e-4" or "1e-4,5e-5"')

    # Baseline toggle
    p.add_argument('--baseline', type=str, choices=['mean','rollout'], default=None, help='Baseline type override')
    p.add_argument('--baseline-eval-batches', type=int, default=None, help='Rollout baseline: eval batches')
    p.add_argument('--baseline-update-frequency', type=int, default=None, help='Rollout baseline: update frequency (epochs)')

    # Optimizer experiment toggles
    p.add_argument('--override-beta2', type=str, default=None, help='true/false: enable AdamW beta2 scaling')
    p.add_argument('--base-beta2', type=float, default=None, help='Base beta2 value for scaling (default 0.999)')
    p.add_argument('--base-batch-size', type=int, default=None, help='Reference batch size for beta2 scaling (default 512)')
    p.add_argument('--normalize-advantages', type=str, choices=['batch','none'], default=None, help='Advantage normalization method')
    p.add_argument('--opt-eps', type=float, default=None, help='Epsilon for advantage normalization')

    return p.parse_args()


def main():
    args = parse_args()

    # Save current directory and change to project root for config loading
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    config_path = Path(original_cwd) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(str(config_path))
    os.chdir(original_cwd)

    logger = setup_logging(config)
    logger.info('='*60)
    logger.info('GT EXPERIMENT DRIVER')
    logger.info('='*60)
    logger.info(f"Base configuration: {args.config}")

    # Base output dir
    if args.output_dir:
        base_dir = Path(args.output_dir)
    elif 'working_dir_path' in config:
        working_dir = Path(config['working_dir_path'])
        base_dir = (project_root / working_dir) if not working_dir.is_absolute() else working_dir
    else:
        base_dir = Path(__file__).parent / 'results_gt'
    base_dir = base_dir.resolve()
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Base output directory: {base_dir}")

    # Device and seeds
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Parse override vectors
    H = parse_list(args.heads, int)
    L = parse_list(args.layers, int)
    D = parse_list(args.hidden, int)
    DR = parse_list(args.dropout, float)
    FF = parse_list(args.ff_mult, int)
    BS = parse_list(args.batch_size, int)
    EPOCHS = parse_list(args.epochs, int)
    LR = parse_list(args.lr, float)

    # If no lists provided, run single experiment using None placeholders
    if not any([H, L, D, DR, FF, BS, EPOCHS, LR]):
        H = [None]
        L = [None]
        D = [None]
        DR = [None]
        FF = [None]
        BS = [None]
        EPOCHS = [None]
        LR = [None]

    planned_runs: List[Tuple[Dict[str, Any], str]] = []

    # Build grid
    for h, l, d, dr, ff, bs, e, lr in product(H, L, D, DR, FF, BS, EPOCHS, LR):
        label = combo_label(h, l, d, dr, ff, bs, lr, e,
                            (None if args.override_beta2 is None else (str(args.override_beta2).lower() == 'true')),
                            args.normalize_advantages)
        run_dir = build_run_dir(base_dir, args.name, label)

        # Build overrides dict for this run
        overrides: Dict[str, Any] = {}
        ensure_sections(config)

        if h is not None:
            overrides.setdefault('model', {})['num_heads'] = h
        if l is not None:
            overrides.setdefault('model', {})['num_layers'] = l
        if d is not None:
            overrides.setdefault('model', {})['hidden_dim'] = d
        if dr is not None:
            overrides.setdefault('model', {})['transformer_dropout'] = dr
        if ff is not None:
            overrides.setdefault('model', {})['feedforward_multiplier'] = ff

        if bs is not None:
            overrides.setdefault('training', {})['batch_size'] = bs
        if e is not None:
            overrides.setdefault('training', {})['num_epochs'] = e
        if lr is not None:
            overrides.setdefault('training', {})['learning_rate'] = lr

        # Baseline overrides
        if args.baseline is not None:
            overrides.setdefault('baseline', {})['type'] = args.baseline
        if args.baseline_eval_batches is not None:
            overrides.setdefault('baseline', {})['eval_batches'] = int(args.baseline_eval_batches)
        if args.baseline_update_frequency is not None:
            overrides.setdefault('baseline', {}).setdefault('update', {})['frequency'] = int(args.baseline_update_frequency)

        # Optimizer experiment toggles
        if any(x is not None for x in [args.override_beta2, args.base_beta2, args.base_batch_size, args.normalize_advantages, args.opt_eps]):
            oe = overrides.setdefault('optimizer_experiments', {})
            if args.override_beta2 is not None:
                oe['override_beta2'] = (str(args.override_beta2).lower() == 'true')
            if args.base_beta2 is not None:
                oe['base_beta2'] = float(args.base_beta2)
            if args.base_batch_size is not None:
                oe['base_batch_size'] = int(args.base_batch_size)
            if args.normalize_advantages is not None:
                oe['normalize_advantages'] = str(args.normalize_advantages)
            if args.opt_eps is not None:
                oe['eps'] = float(args.opt_eps)

        planned_runs.append((overrides, str(run_dir)))

    # Dry run preview
    if args.dry_run:
        logger.info('[GT-EXP] Dry run: planned configurations:')
        for ov, rd in planned_runs:
            logger.info(f' - run_dir={rd}\n   overrides={json.dumps(ov, indent=2)}')
        return

    # Train runs
    for overrides, run_dir in planned_runs:
        run_path = Path(run_dir)
        if run_path.exists() and not args.force_retrain:
            logger.info(f"[GT-EXP] Skipping existing run (use --force-retrain to overwrite): {run_dir}")
            continue

        # Prepare run directory
        os.makedirs(run_dir, exist_ok=True)

        # Compose run config
        run_cfg = apply_overrides(config, overrides)
        ensure_sections(run_cfg)

        # Log effective config snapshot for reproducibility
        with open(os.path.join(run_dir, 'effective_config.json'), 'w') as f:
            json.dump(run_cfg, f, indent=2)

        # Data generator bound to run config
        data_generator = create_data_generator(run_cfg)

        # Model
        model_name = 'GT+RL'
        model = ModelFactory.create_model(model_name, run_cfg)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info('-'*40)
        logger.info(f"[GT-EXP] Run: {run_dir}")
        logger.info(f"[GT-EXP] Model: {model.__class__.__name__}, params: {total_params:,}")

        csv_writer = IncrementalCSVWriter(run_dir, model_name, run_cfg, logger)

        # Train with experimental trainer
        start_time = time.time()
        history, training_time, artifacts = advanced_train_model_opt(
            model=model,
            model_name=model_name,
            config=run_cfg,
            data_generator=data_generator,
            logger_print=logger.info,
            use_advanced_features=run_cfg.get('experiment', {}).get('use_advanced_features', True),
            epoch_callback=csv_writer.write_epoch
        )
        csv_writer = None  # allow early GC of buffers

        results = {'history': history, 'artifacts': artifacts}
        save_model(run_dir, model_name, model, results, training_time, run_cfg, logger)

        # Summary
        final_val = history.get('final_val_cost', float('inf'))
        best_val = artifacts.get('best_val_cost', final_val)
        convergence_epoch = artifacts.get('convergence_epoch', 'N/A')
        logger.info(f"[GT-EXP] Complete: time={training_time:.1f}s, final_val={final_val:.4f}, best_val={best_val:.4f}, conv_epoch={convergence_epoch}")


if __name__ == '__main__':
    main()
