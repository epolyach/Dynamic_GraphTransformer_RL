#!/usr/bin/env python3
"""
Build enhanced_comparative_study.pt from saved model checkpoints.

This script scans the working_dir_path (from --config) for saved models under
<prefix>/pytorch/model_*.pt, extracts per-model training histories and times,
and writes a consolidated analysis file to <prefix>/analysis/enhanced_comparative_study.pt.

Usage:
  python3 build_analysis_artifact.py --config configs/small.yaml
  # optional override of output base dir (otherwise reads from config)
  python3 build_analysis_artifact.py --config configs/small.yaml --base-dir results/small
"""

import argparse
from pathlib import Path
import torch
import sys


def load_config(config_path: str) -> dict:
    from src.utils.config import load_config as _load
    return _load(config_path)


def parse_args():
    p = argparse.ArgumentParser(description="Build enhanced comparative analysis artifact from saved models")
    p.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    p.add_argument("--base-dir", type=str, default=None,
                   help="Override working_dir_path from config (e.g., results/small)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    base_dir = Path(args.base_dir or cfg.get("working_dir_path", "results"))
    pytorch_dir = base_dir / "pytorch"
    analysis_dir = base_dir / "analysis"

    if not pytorch_dir.exists():
        print(f"ERROR: Models directory not found: {pytorch_dir}", file=sys.stderr)
        sys.exit(1)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    training_times = {}
    config = None

    model_files = sorted(pytorch_dir.glob("model_*.pt"))
    if not model_files:
        print(f"ERROR: No model_*.pt files found in {pytorch_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(model_files)} model files in {pytorch_dir}")

    for f in model_files:
        try:
            m = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ⚠️  Failed to load {f.name}: {e}")
            continue
        name = m.get("model_name") or f.stem.replace("model_", "").replace("_", " ").replace("plus", "+")
        hist = m.get("history", {})
        if not hist:
            print(f"  ⚠️  No history found in {f.name}; skipping")
            continue
        results[name] = {"history": hist}
        training_times[name] = float(m.get("training_time", 0.0))
        if config is None:
            config = m.get("config", cfg)
        print(f"  ✓ {name}: epochs={len(hist.get('train_costs', []))}, time={training_times[name]:.1f}s")

    if not results:
        print("ERROR: No valid model histories were found.", file=sys.stderr)
        sys.exit(1)

    out_path = analysis_dir / "enhanced_comparative_study.pt"
    torch.save({
        "results": results,
        "training_times": training_times,
        "config": config,
    }, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
