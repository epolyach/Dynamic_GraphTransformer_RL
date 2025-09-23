#!/usr/bin/env python3
"""Test the Epoch Execution Time (EET) with optimized baseline frequency"""
import time
import sys
import os
import yaml
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

# Create test config with 3 epochs to see the pattern
test_config = {
    'working_dir_path': '../results/test_eet',
    'problem': {
        'num_customers': 10,
        'vehicle_capacity': 30
    },
    'training': {
        'num_batches_per_epoch': 25,
        'batch_size': 128,
        'num_epochs': 6  # 6 epochs to see baseline updates at 0, 3, 6
    },
    'model': {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 3
    },
    'baseline': {
        'eval_batches': 2,  # Reduced from 5
        'update': {
            'frequency': 3,  # Every 3 epochs
            'warmup_epochs': 0
        }
    }
}

# Write test config
with open('configs/test_eet.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("="*70)
print("EPOCH EXECUTION TIME (EET) TEST")
print("="*70)
print("Configuration:")
print("  - Problem: n=10, capacity=30")
print("  - Training: 25 batches × 128 instances = 3,200 per epoch")
print("  - Baseline: evaluate every 3 epochs (epochs 0, 3, 6)")
print("  - Baseline eval dataset: 2 batches (reduced from 5)")
print("="*70)

# Run training
original_argv = sys.argv
sys.argv = ['run_training_gpu.py',
            '--config', 'configs/test_eet.yaml',
            '--model', 'GT+RL',
            '--device', 'cuda:0',
            '--force-retrain']

try:
    from training_gpu.scripts.run_training_gpu import main as gpu_main
    
    print("\nStarting training...\n")
    start_time = time.time()
    gpu_main()
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  Total time for 6 epochs: {total_time:.2f} seconds")
    print(f"  Average per epoch: {total_time/6:.2f} seconds")
    print("\nExpected pattern:")
    print("  - Epochs 0, 3: slower (with baseline evaluation)")
    print("  - Epochs 1, 2, 4, 5: faster (no baseline evaluation)")
    print("\nTarget EET: 22-23 seconds for epochs WITHOUT baseline")
    print("="*70)
    
finally:
    sys.argv = original_argv
    # Clean up
    if os.path.exists('configs/test_eet.yaml'):
        os.remove('configs/test_eet.yaml')
