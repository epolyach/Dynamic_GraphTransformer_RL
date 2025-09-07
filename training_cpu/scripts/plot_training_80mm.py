#!/usr/bin/env python3
"""
Plot training curves (CPC vs Epoch) for CVRP models.
Creates publication-quality 80mm single-column figures.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def resolve_config_path(scripts_dir: Path, cfg: str) -> Path:
    candidate = scripts_dir.parent.parent / 'configs' / cfg
    if candidate.exists():
        return candidate
    p = Path(cfg)
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(f"Config not found: {cfg} (tried {candidate} and {p})")


def load_working_dir(config_path: Path) -> Path:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    wd = Path(cfg.get('working_dir_path', 'training_cpu/results/tiny'))
    if not wd.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        wd = (repo_root / wd).resolve()
    return wd


def main():
    parser = argparse.ArgumentParser(description='Plot training curves (CPC vs Epoch), 80mm single column')
    parser.add_argument('--config', required=True, help='YAML config name or path (e.g., tiny.yaml)')
    parser.add_argument('--name', required=True, help='Output base name (e.g., tiny_n10)')
    parser.add_argument('--level', type=float, default=None, help='Horizontal reference line level (e.g., 0.515)')
    parser.add_argument('--comment', type=str, default=None, help='Optional yellow comment box text')
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    config_path = resolve_config_path(scripts_dir, args.config)
    working_dir = load_working_dir(config_path)

    csv_dir = working_dir / 'csv'
    plots_dir = working_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Figure: 80mm single-column
    width_in = mm_to_inches(80.0)
    height_in = width_in / 1.618
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Models to plot (training cost only)
    models = [
        ('history_gat_rl.csv', 'GAT+RL', '#1f77b4'),
        ('history_gt_rl.csv',  'GT+RL',  '#ff7f0e'),
        ('history_dgt_rl.csv', 'DGT+RL', '#2ca02c'),
    ]

    for fname, label, color in models:
        path = csv_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if 'epoch' not in df.columns or 'train_cost' not in df.columns:
            continue
        ax.plot(df['epoch'], df['train_cost'], label=label, color=color, linewidth=0.8)

    # Optional horizontal level
    if args.level is not None:
        ax.axhline(args.level, color='gray', linestyle='--', linewidth=0.6, alpha=0.8)

    # Optional comment box
    if args.comment:
        ax.text(
            0.5, 0.99, args.comment, transform=ax.transAxes,
            fontsize=5, va='top', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='goldenrod', alpha=0.5, linewidth=0.5)
        )

    # Styling
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('CPC', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(loc='best', fontsize=5, frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_base = plots_dir / args.name
    fig.savefig(f"{out_base}.eps", format='eps', dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_base}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_base}.eps and {out_base}.png")


if __name__ == '__main__':
    main()

