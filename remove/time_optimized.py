#!/usr/bin/env python3
import time,sys,os
sys.path.insert(0,'/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
import yaml
test_config=yaml.safe_load(open('configs/tiny_gpu_512.yaml'))
test_config['training']['num_epochs']=2
test_config['training']['num_batches_per_epoch']=50  # quick test
test_config['training_advanced']={'compile':{'enabled':True,'mode':'default'}}
with open('configs/test_optimized.yaml','w') as f: yaml.dump(test_config,f)
sys.argv=['run','--config','configs/test_optimized.yaml','--model','GT+RL','--device','cuda:0','--force-retrain']
from training_gpu.scripts.run_training_gpu import main as gpu_main
start=time.time()
gpu_main()
elapsed=time.time()-start
print(f"\n{'='*60}")
print(f"Test Results (50 batches × 512):")
print(f"  Total time: {elapsed:.1f}s for 2 epochs")
print(f"  Per-epoch: {elapsed/2:.1f}s")
print(f"  Throughput: {(50*512*2)/elapsed:.0f} instances/second")
print(f"{'='*60}")
if os.path.exists('configs/test_optimized.yaml'):os.remove('configs/test_optimized.yaml')
