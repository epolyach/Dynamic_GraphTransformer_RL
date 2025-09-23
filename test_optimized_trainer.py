#!/usr/bin/env python3
"""Test the optimized trainer to verify 22-second epoch times"""
import time
import sys
import os
import yaml
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

# Create a test config for 1 epoch
test_config = {
    'working_dir_path': '../results/test_optimized',
    'problem': {
        'num_customers': 10,
        'vehicle_capacity': 30
    },
    'training': {
        'num_batches_per_epoch': 25,
        'batch_size': 128,
        'num_epochs': 1  # Just 1 epoch for timing
    },
    'model': {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 3
    },
    'model_gat': {
        'gat_hidden_dim': 128,
        'gat_layers': 3
    }
}

# Write test config
with open('configs/test_timing.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("="*60)
print("Testing optimized trainer with n=10, capacity=30")
print("Configuration: 25 batches x 128 batch_size = 3200 instances")
print("="*60)

# Mock sys.argv for the training script
original_argv = sys.argv
sys.argv = ['run_training_gpu.py', 
            '--config', 'configs/test_timing.yaml',
            '--model', 'GT+RL', 
            '--device', 'cuda:0',
            '--force-retrain']

start_time = time.time()

try:
    # Import and run
    from training_gpu.scripts.run_training_gpu import main as gpu_main
    gpu_main()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"RESULTS:")
    print(f"  Time for 1 epoch: {elapsed:.2f} seconds")
    print(f"  Target: ~22-23 seconds")
    if elapsed < 25:
        print(f"  Status: ✓ SUCCESS - Performance target achieved!")
    else:
        print(f"  Status: ✗ SLOW - Need further optimization")
    print("="*60)
    
finally:
    sys.argv = original_argv
    # Clean up
    if os.path.exists('configs/test_timing.yaml'):
        os.remove('configs/test_timing.yaml')
