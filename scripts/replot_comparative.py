#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.comprehensive_plots import (
    load_comprehensive_data_for_plots,
    create_comprehensive_training_plots,
)


def main():
    p = argparse.ArgumentParser(description='Replot comprehensive comparison from existing CSV and histories (no retraining)')
    p.add_argument('--csv', type=str, default='results/comparative_study_cpu.csv', help='Path to consolidated results CSV')
    p.add_argument('--results_dir', type=str, default='results_train', help='Base directory with per-model run folders')
    p.add_argument('--customers', type=int, default=20, help='Number of customers used in the runs')
    p.add_argument('--out', type=str, default='utils/plots/comparative_study_results.png', help='Output plot path')
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)

    results = load_comprehensive_data_for_plots(df, Path(args.results_dir))
    out_path = Path(args.out)
    create_comprehensive_training_plots(results, out_path, customers=args.customers)

    print(f'Regenerated plot at {out_path}')


if __name__ == '__main__':
    main()
