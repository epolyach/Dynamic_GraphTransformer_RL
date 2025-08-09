#!/usr/bin/env python3
from src.training.pipeline_train import main as unified_main
import argparse
import sys

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train CVRP model on GPU with AMP')
    ap.add_argument('--model', choices=['pointer_rl', 'static_rl', 'dynamic_gt_rl'], default='dynamic_gt_rl')
    ap.add_argument('--customers', type=int, default=20)
    ap.add_argument('--instances', type=int, default=800)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--capacity', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--data_seed', type=int, default=12345)
    ap.add_argument('--out_dir', type=str, default='results_train')
    args = ap.parse_args()

    sys.argv = [sys.argv[0],
                '--pipeline', 'gpu_amp',
                '--model', args.model,
                '--customers', str(args.customers),
                '--instances', str(args.instances),
                '--epochs', str(args.epochs),
                '--batch', str(args.batch),
                '--capacity', str(args.capacity),
                '--lr', str(args.lr),
                '--data_seed', str(args.data_seed),
                '--out_dir', args.out_dir]
    unified_main()
