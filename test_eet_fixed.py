#!/usr/bin/env python3
"""Test EET after fixing baseline evaluation"""
import time
import sys
import os
import yaml
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

test_config = {
    'working_dir_path': '../results/test_eet_fixed',
    'problem': {
        'num_customers': 10,
        'vehicle_capacity': 30
    },
    'training': {
        'num_batches_per_epoch': 25,
        'batch_size': 128,
        'num_epochs': 3  # Just 3 epochs for quick test
    },
    'model': {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 3
    },
    'baseline': {
        'eval_batches': 2,
        'update': {
            'frequency': 3,
            'warmup_epochs': 0
        }
    }
}

with open('configs/test_eet_fixed.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("="*70)
print("EPOCH EXECUTION TIME TEST - AFTER FIX")
print("="*70)
print("Fixed: Baseline no longer evaluates on every batch")
print("Configuration: n=10, capacity=30, 25 batches × 128")
print("="*70)

sys.argv = ['run_training_gpu.py',
            '--config', 'configs/test_eet_fixed.yaml',
            '--model', 'GT+RL',
            '--device', 'cuda:0',
            '--force-retrain']

try:
    from training_gpu.scripts.run_training_gpu import main as gpu_main
    
    print("\nStarting training with fixed baseline...\n")
    start_time = time.time()
    gpu_main()
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("RESULTS AFTER FIX:")
    print(f"  Total time for 3 epochs: {total_time:.2f} seconds")
    print(f"  Average per epoch: {total_time/3:.2f} seconds")
    print(f"\n  Target: 22-23 seconds per epoch")
    avg = total_time/3
    if avg <= 25:
        print(f"  ✓ SUCCESS! Achieved {avg:.1f}s per epoch")
    else:
        print(f"  Still needs work: {avg:.1f}s per epoch")
    print("="*70)
    
finally:
    sys.argv = sys.argv[:1]
    if os.path.exists('configs/test_eet_fixed.yaml'):
        os.remove('configs/test_eet_fixed.yaml')
