#!/usr/bin/env python3
"""Minimal test to verify rollout baseline configuration and logic."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Check config loading and baseline settings
print("=" * 60)
print("TEST 1: Configuration Check")
print("=" * 60)

from src.utils.config import load_config

config = load_config("configs/small_quick.yaml")
baseline_cfg = config.get('baseline', {})

print(f"Baseline type: {baseline_cfg.get('type', 'NOT SET')}")
print(f"Baseline config: {baseline_cfg}")

if baseline_cfg.get('type') == 'rollout':
    print("✅ Rollout baseline is ENABLED in config")
    print(f"  - Eval batches: {baseline_cfg.get('eval_batches', 2)}")
    print(f"  - Greedy temperature: {baseline_cfg.get('greedy_temperature', 0.1)}")
    update_cfg = baseline_cfg.get('update', {})
    print(f"  - Update enabled: {update_cfg.get('enabled', True)}")
    print(f"  - Update frequency: {update_cfg.get('frequency', 1)}")
    print(f"  - Significance test: {update_cfg.get('significance_test', True)}")
    print(f"  - P-value threshold: {update_cfg.get('p_value', 0.05)}")
else:
    print("❌ Rollout baseline is NOT enabled (using mean baseline)")

# Test 2: Check that rollout baseline module exists and imports
print("\n" + "=" * 60)
print("TEST 2: Module Import Check")
print("=" * 60)

try:
    from src.training.rollout_baseline import RolloutBaseline
    print("✅ RolloutBaseline class imported successfully")
    
    # Check required methods
    required_methods = ['eval_batch', 'epoch_callback', '_update_model', '_compute_batch_costs']
    for method in required_methods:
        if hasattr(RolloutBaseline, method):
            print(f"  ✅ Method '{method}' found")
        else:
            print(f"  ❌ Method '{method}' missing")
except ImportError as e:
    print(f"❌ Failed to import RolloutBaseline: {e}")

# Test 3: Check advanced trainer integration
print("\n" + "=" * 60)
print("TEST 3: Advanced Trainer Integration Check")
print("=" * 60)

try:
    # Read the advanced_trainer.py file to verify integration
    trainer_path = "src/training/advanced_trainer.py"
    with open(trainer_path, 'r') as f:
        trainer_code = f.read()
    
    # Check for key integration points
    checks = [
        ("Import statement", "from src.training.rollout_baseline import RolloutBaseline"),
        ("Baseline setup", "use_rollout_baseline = str(baseline_cfg.get('type', 'mean')).lower() == 'rollout'"),
        ("Baseline instantiation", "baseline = RolloutBaseline(model, eval_dataset, config, logger_print)"),
        ("Baseline evaluation", "bl_vals = baseline.eval_batch(instances)"),
        ("Baseline callback", "baseline.epoch_callback(model, epoch)"),
    ]
    
    for check_name, code_snippet in checks:
        if code_snippet in trainer_code:
            print(f"  ✅ {check_name}: Found")
        else:
            print(f"  ❌ {check_name}: Not found")
            
except FileNotFoundError:
    print(f"❌ Could not find {trainer_path}")
except Exception as e:
    print(f"❌ Error checking trainer integration: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if baseline_cfg.get('type') == 'rollout':
    print("✅ Rollout baseline is configured and should be used during training")
    print("   To disable it, set baseline.type to 'mean' in your config")
else:
    print("❌ Rollout baseline is NOT configured")
    print("   To enable it, set baseline.type to 'rollout' in your config")

print("\nNote: When running actual training with rollout baseline enabled,")
print("you should see log messages like:")
print("  [RolloutBaseline] Initialized at epoch 0 with mean=X.XXXXXX")
print("  [RolloutBaseline] Evaluating candidate on eval dataset...")
print("  [RolloutBaseline] Epoch N: candidate mean=X.XX, baseline mean=Y.YY")
