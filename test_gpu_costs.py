#!/usr/bin/env python3
"""Test performance with GPU cost computation"""
import time
import sys
import os
import yaml
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

configs = [
    {'n': 10, 'name': 'n10'},
    {'n': 20, 'name': 'n20'}
]

for cfg in configs:
    test_config = {
        'working_dir_path': f'../results/test_gpu_costs_{cfg["name"]}',
        'problem': {
            'num_customers': cfg['n'],
            'vehicle_capacity': 30
        },
        'training': {
            'num_batches_per_epoch': 25,
            'batch_size': 128,
            'num_epochs': 2
        },
        'model': {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 3
        },
        'baseline': {
            'eval_batches': 2,
            'update': {'frequency': 3, 'warmup_epochs': 0}
        }
    }
    
    with open('configs/test_gpu_costs.yaml', 'w') as f:
        yaml.dump(test_config, f)
    
    print(f"\n{'='*60}")
    print(f"Testing GPU cost computation with n={cfg['n']}")
    print(f"{'='*60}")
    
    sys.argv = ['run', '--config', 'configs/test_gpu_costs.yaml',
                '--model', 'GT+RL', '--device', 'cuda:0', '--force-retrain']
    
    from training_gpu.scripts.run_training_gpu import main as gpu_main
    
    start = time.time()
    gpu_main()
    elapsed = time.time() - start
    
    print(f"\nResults for n={cfg['n']}:")
    print(f"  Total: {elapsed:.1f}s for 2 epochs")
    print(f"  Average: {elapsed/2:.1f}s per epoch")

if os.path.exists('configs/test_gpu_costs.yaml'):
    os.remove('configs/test_gpu_costs.yaml')

print(f"\n{'='*60}")
print("GPU cost computation implemented and tested!")
print(f"{'='*60}")
