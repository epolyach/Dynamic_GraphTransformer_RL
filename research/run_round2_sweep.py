#!/usr/bin/env python3
"""
Round 2 model-aware hyperparameter sweep centered on GAT+RL best.
- Validates divisibility constraints (embed_dim % n_heads == 0)
- Avoids unusable, model-specific knobs (no GAT-only params sent to GT/DGT)
- Performs a GAT marginality check via one-at-a-time perturbations
- Uses early-stopping trainer script for efficiency
- Aggregates results across seeds; writes CSV and quick plots
"""

import os
import sys
import json
import time
import math
import copy
import yaml
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CONFIG_PATH = 'configs/round2_sweep.yaml'
TRAINER = 'run_experimental_training_with_early_stopping.py'


def log(msg: str):
    print(msg, flush=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def divisible_heads(embed_dim: int, n_heads: int) -> bool:
    return embed_dim % n_heads == 0


def nearest_valid_heads(embed_dim: int, target_heads: int) -> int:
    candidates = [h for h in range(2, min(32, embed_dim) + 1) if embed_dim % h == 0]
    if not candidates:
        return max(1, min(target_heads, embed_dim))
    return min(candidates, key=lambda h: abs(h - target_heads))


def filter_conditionals(model: str, conditionals: Dict[str, Any]) -> Dict[str, List[Any]]:
    usable = {}
    for name, spec in (conditionals or {}).items():
        models = spec.get('models', [])
        if model in models:
            usable[name] = spec.get('values', [])
    return usable


def build_trials(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep = cfg['sweep']
    models = sweep['models']
    seeds = sweep.get('seeds', [42])
    grid = sweep['grid']
    conditionals = sweep.get('conditionals', {})

    # Build Cartesian grid for base params
    base_keys = ['embedding_dim', 'n_layers', 'n_heads', 'learning_rate', 'dropout', 'batch_size', 'temp_start', 'temp_min', 'temp_decay']
    base_values = [grid[k] for k in base_keys]
    base_combos = list(itertools.product(*base_values))

    trials = []
    for model in models:
        extra = filter_conditionals(model, conditionals)
        extra_keys = list(extra.keys())
        extra_values = [extra[k] for k in extra_keys]
        extra_combos = list(itertools.product(*extra_values)) if extra_keys else [()]

        for combo in base_combos:
            combo_dict = dict(zip(base_keys, combo))
            # Enforce valid heads
            if not divisible_heads(int(combo_dict['embedding_dim']), int(combo_dict['n_heads'])):
                # try to fix
                fixed_heads = nearest_valid_heads(int(combo_dict['embedding_dim']), int(combo_dict['n_heads']))
                if fixed_heads < 2 or (int(combo_dict['embedding_dim']) % fixed_heads != 0):
                    continue
                combo_dict['n_heads'] = fixed_heads

            for extra_combo in extra_combos:
                params = combo_dict.copy()
                params.update(dict(zip(extra_keys, extra_combo)))
                for seed in seeds:
                    trial = {
                        'model': model,
                        'seed': int(seed),
                        'params': params
                    }
                    trials.append(trial)
    return trials


def build_gat_marginality_trials(cfg: Dict[str, Any], best_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    m = cfg['sweep'].get('gat_marginality', {})
    if not m or not m.get('enable', True):
        return []

    lr_scale = m.get('lr_scale', [1.0])
    entropy_scale = m.get('entropy_scale', [1.0])
    batch_opts = m.get('batch_options', [best_cfg['batch_size']])
    grad_opts = m.get('grad_clip_options', [1.5])
    attn_drop_opts = m.get('attn_dropout_options', [0.1, 0.2])
    n_layers_delta = m.get('n_layers_delta', [0])
    tanh_opts = m.get('tanh_clip_options', [15])

    seeds = cfg['sweep'].get('seeds', [42])

    trials = []
    for ls, es, bs, gc, ad, nd, tc in itertools.product(lr_scale, entropy_scale, batch_opts, grad_opts, attn_drop_opts, n_layers_delta, tanh_opts):
        cfg2 = copy.deepcopy(best_cfg)
        cfg2['learning_rate'] = float(best_cfg['learning_rate']) * float(ls)
        cfg2['entropy_coef'] = float(cfg.get('training_advanced', {}).get('entropy_coef', 0.003)) * float(es)
        cfg2['batch_size'] = int(bs)
        cfg2['gradient_clip_norm'] = float(gc)
        cfg2['attention_dropout'] = float(ad)
        cfg2['n_layers'] = max(2, int(best_cfg['n_layers']) + int(nd))
        cfg2['tanh_clipping'] = int(tc)

        # Keep n_heads valid
        if not divisible_heads(int(cfg2['embedding_dim']), int(cfg2['n_heads'])):
            cfg2['n_heads'] = nearest_valid_heads(int(cfg2['embedding_dim']), int(cfg2['n_heads']))

        for seed in seeds:
            trials.append({'model': 'GAT+RL', 'seed': int(seed), 'params': cfg2})
    return trials


def run_trial(trial: Dict[str, Any], base_cfg: Dict[str, Any], config_path: str) -> Tuple[float, Dict[str, Any], str]:
    model = trial['model']
    params = trial['params']
    seed = trial['seed']

    # Build command
    cmd = [
        'python', TRAINER,
        '--config', config_path,
        '--models', model,
        '--embedding_dim', str(params['embedding_dim']),
        '--n_layers', str(params['n_layers']),
        '--n_heads', str(params['n_heads']),
        '--learning_rate', str(params['learning_rate']),
        '--dropout', str(params['dropout']),
        '--batch_size', str(params['batch_size']),
        '--temp_start', str(params['temp_start']),
        '--temp_min', str(params['temp_min']),
        '--temp_decay', str(params['temp_decay']),
        '--early_stopping_patience', '36',
        '--min_epochs', '16',
        '--max_epochs', str(base_cfg['training']['num_epochs'])
    ]

    # Optional params if present in this trial
    optional = {
        'distance_encoding_strength': '--distance_encoding_strength',
        'dynamic_state_updates_per_decode': '--dynamic_state_updates_per_decode',
        'state_update_type': '--state_update_type',
        'entropy_coef': '--entropy_coef',               # early-stopping script reads from config; include if supported
        'gradient_clip_norm': '--gradient_clip_norm',   # same note
        'attention_dropout': '--attention_dropout',     # include if model supports
        'tanh_clipping': '--tanh_clipping',
    }

    for k, flag in optional.items():
        if k in params:
            cmd.extend([flag, str(params[k])])

    # Environment: set seed via config random_seed override
    # The trainer uses set_seeds(config['experiment']['random_seed'])
    # We pass via ENV var and the script will keep using config; we also fallback to override in config if needed.
    env = os.environ.copy()
    env['SEARCH_RANDOM_SEED'] = str(seed)

    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start

    val_cost = float('inf')
    epochs_trained = None
    # Parse stdout for final validation cost
    for line in res.stdout.strip().split('\n'):
        if line.lower().startswith('final validation cost') or 'final validation cost:' in line.lower():
            try:
                val_cost = float(line.split(':')[-1].strip())
            except Exception:
                pass
        if line.lower().startswith('epochs trained') or 'epochs trained:' in line.lower():
            try:
                epochs_trained = int(line.split(':')[-1].strip())
            except Exception:
                pass

    return val_cost, {
        'stdout': res.stdout[-2000:],
        'stderr': res.stderr[-2000:],
        'returncode': res.returncode,
        'elapsed_sec': elapsed,
        'epochs_trained': epochs_trained,
    }, 'OK' if res.returncode == 0 else 'FAIL'


def summarize_and_plot(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'round2_results.csv'), index=False)

    if len(df) == 0:
        return

    plt.style.use('seaborn-v0_8')
    # Best per model
    best = df.groupby('model')['val_cost'].min().sort_values()
    plt.figure(figsize=(7,4))
    best.plot(kind='bar', alpha=0.8)
    plt.ylabel('Validation cost')
    plt.title('Round 2: Best per model')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'best_per_model.png'), dpi=200)
    plt.close()

    # LR vs cost scatter
    plt.figure(figsize=(6,4))
    for m in df['model'].unique():
        sub = df[df['model'] == m]
        plt.scatter(sub['learning_rate'], sub['val_cost'], label=m, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Validation cost')
    plt.title('LR vs cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lr_vs_cost.png'), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Round 2 sweep runner')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to sweep YAML')
    parser.add_argument('--max_trials', type=int, default=None, help='Optional cap on number of trials to run')
    parser.add_argument('--skip_marginality', action='store_true', help='Skip GAT marginality trials')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg.get('working_dir_path', 'results/round2_sweep')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build base trials
    trials = build_trials(cfg)
    log(f"Planned base trials: {len(trials)}")

    # Establish a GAT-centered best guess using the middle of grids (or config defaults)
    grid = cfg['sweep']['grid']
    best_guess = {
        'embedding_dim': grid['embedding_dim'][0],
        'n_layers': grid['n_layers'][1] if len(grid['n_layers']) > 1 else grid['n_layers'][0],
        'n_heads': grid['n_heads'][0],
        'learning_rate': grid['learning_rate'][-1],
        'dropout': grid['dropout'][1] if len(grid['dropout']) > 1 else grid['dropout'][0],
        'batch_size': grid['batch_size'][1] if len(grid['batch_size']) > 1 else grid['batch_size'][0],
        'temp_start': grid['temp_start'][0],
        'temp_min': grid['temp_min'][0],
        'temp_decay': grid['temp_decay'][0],
        # Defaults for optional knobs used in marginality
        'entropy_coef': cfg.get('training_advanced', {}).get('entropy_coef', 0.003),
        'gradient_clip_norm': cfg.get('training_advanced', {}).get('gradient_clip_norm', 1.5),
        'attention_dropout': 0.1,
        'tanh_clipping': 15,
    }

    # GAT marginality trials
    m_trials = [] if args.skip_marginality else build_gat_marginality_trials(cfg, best_guess)
    log(f"Planned GAT marginality trials: {len(m_trials)}")

    all_trials = trials + m_trials

    # Apply max_trials cap if requested
    if args.max_trials is not None:
        all_trials = all_trials[:max(0, args.max_trials)]
        log(f"Capped to first {len(all_trials)} trials via --max_trials")

    # Execute trials
    rows = []
    for idx, t in enumerate(all_trials, 1):
        log(f"[{idx}/{len(all_trials)}] {t['model']} seed={t['seed']} params={t['params']}")
        val, meta, status = run_trial(t, cfg, args.config)
        row = {
            'model': t['model'],
            'seed': t['seed'],
            'val_cost': val,
            'status': status,
            'elapsed_sec': meta['elapsed_sec'],
        }
        row.update(t['params'])
        rows.append(row)

        # Save incremental
        df_inc = pd.DataFrame(rows)
        df_inc.to_csv(os.path.join(out_dir, 'round2_results_incremental.csv'), index=False)

    df = pd.DataFrame(rows)
    summarize_and_plot(df, out_dir)

    # Print quick summary
    if len(df) > 0:
        log("\nTop 10 trials by val_cost:")
        top = df.nsmallest(10, 'val_cost')
        for i, r in enumerate(top.itertuples(), 1):
            log(f"{i:2d}. {r.model:6} val={r.val_cost:.4f} emb={r.embedding_dim} L={r.n_layers} H={r.n_heads} lr={r.learning_rate:.1e} drop={r.dropout} bs={r.batch_size}")


if __name__ == '__main__':
    main()

