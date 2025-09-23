#!/usr/bin/env python
"""Quick test to measure training speed after fix"""
import time
import sys
import os
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from training_gpu.scripts.run_training_gpu import main

# Override config for quick test
test_config = {
    'config': 'configs/tiny_1.yaml',
    'model': 'GT+RL',
    'device': 'cuda:0',
    'force_retrain': True,
    'n_epochs_override': 1,  # Just test 1 epoch
}

print("Testing training speed with fixed advanced_trainer_gpu.py...")
print("Running 1 epoch to measure time per epoch...")
start = time.time()

# Mock sys.argv for the main function
original_argv = sys.argv
sys.argv = ['run_training_gpu.py', '--config', 'configs/tiny_1.yaml', '--model', 'GT+RL', '--device', 'cuda:0']

try:
    # We'll need to modify the config to run just 1 epoch
    import yaml
    with open('configs/tiny_1.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily modify for 1 epoch test
    original_epochs = config['training']['num_epochs']
    config['training']['num_epochs'] = 1
    
    with open('configs/tiny_1_temp.yaml', 'w') as f:
        yaml.dump(config, f)
    
    sys.argv = ['run_training_gpu.py', '--config', 'configs/tiny_1_temp.yaml', '--model', 'GT+RL', '--device', 'cuda:0', '--force-retrain']
    
    # Import and run
    from training_gpu.scripts.run_training_gpu import main as gpu_main
    gpu_main()
    
    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Time for 1 epoch: {elapsed:.2f} seconds")
    print(f"Expected time for 10 epochs: {elapsed*10:.2f} seconds")
    print(f"{'='*50}")
    
finally:
    sys.argv = original_argv
    # Clean up temp config
    if os.path.exists('configs/tiny_1_temp.yaml'):
        os.remove('configs/tiny_1_temp.yaml')
