#!/usr/bin/env python3
"""
Production runner for OR-Tools GLS benchmarks
Runs 10,000 instances per configuration with checkpointing
"""

import os
import sys
import time
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import signal
import glob

# Global state for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n[{datetime.now()}] Shutdown requested. Finishing current batch...")
    shutdown_requested = True

def load_checkpoint(checkpoint_file):
    """Load progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_file, state):
    """Save progress to checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=2)

def run_batch(config, batch_size=100, start_idx=0):
    """Run a batch of instances"""
    n, capacity, timeout, output_dir = config
    
    # Use virtual environment Python
    venv_python = os.path.abspath('../venv/bin/python3')
    
    cmd = [
        venv_python, 'scripts/benchmark_ortools_gls_fixed.py',
        '--n', str(n),
        '--capacity', str(capacity),
        '--instances', str(batch_size),
        '--timeout', str(timeout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=batch_size * timeout + 60)  # Add buffer
        
        if result.returncode == 0:
            # Move JSON file to output directory
            json_files = [f for f in os.listdir('.') 
                         if f.startswith(f'ortools_gls_n{n}_') and f.endswith('.json')]
            
            for jf in json_files:
                # Rename with batch index
                new_name = f'batch_{start_idx:05d}_{jf}'
                new_path = os.path.join(output_dir, new_name)
                os.rename(jf, new_path)
                
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Batch timeout"
    except Exception as e:
        return False, str(e)

def run_configuration(config, total_instances=10000, batch_size=100):
    """Run a full configuration with checkpointing"""
    n, capacity, timeout, output_dir = config
    config_id = f"n{n}_timeout{timeout}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoint file
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{config_id}.json')
    checkpoint = load_checkpoint(checkpoint_file)
    
    # Get progress
    completed_batches = checkpoint.get('completed_batches', [])
    failed_batches = checkpoint.get('failed_batches', [])
    
    print(f"\n[{datetime.now()}] Configuration N={n}, timeout={timeout}s")
    print(f"  Progress: {len(completed_batches) * batch_size}/{total_instances} instances")
    
    # Calculate remaining batches
    num_batches = (total_instances + batch_size - 1) // batch_size
    remaining_batches = [i for i in range(num_batches) 
                        if i not in completed_batches and i not in failed_batches]
    
    if not remaining_batches:
        print(f"  Already completed!")
        return len(completed_batches) * batch_size
    
    # Process remaining batches
    for batch_idx in remaining_batches:
        if shutdown_requested:
            print(f"  Stopping at batch {batch_idx}/{num_batches}")
            break
            
        start_idx = batch_idx * batch_size
        actual_batch_size = min(batch_size, total_instances - start_idx)
        
        print(f"  Batch {batch_idx + 1}/{num_batches} ({actual_batch_size} instances)...", end='', flush=True)
        
        success, error = run_batch(
            (n, capacity, timeout, output_dir), 
            actual_batch_size, 
            start_idx
        )
        
        if success:
            print(" ✓")
            completed_batches.append(batch_idx)
        else:
            print(f" ✗ ({error[:50]})")
            failed_batches.append(batch_idx)
            # Retry once
            time.sleep(2)
            print(f"    Retrying batch {batch_idx + 1}...", end='', flush=True)
            success, error = run_batch(
                (n, capacity, timeout, output_dir), 
                actual_batch_size, 
                start_idx
            )
            if success:
                print(" ✓")
                completed_batches.append(batch_idx)
                failed_batches.remove(batch_idx)
            else:
                print(" ✗")
        
        # Update checkpoint
        checkpoint['completed_batches'] = completed_batches
        checkpoint['failed_batches'] = failed_batches
        checkpoint['last_update'] = datetime.now().isoformat()
        save_checkpoint(checkpoint_file, checkpoint)
    
    return len(completed_batches) * batch_size

def main():
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 70)
    print("OR-Tools GLS Production Benchmark Runner")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Target: 10,000 instances per configuration")
    print(f"Batch size: 100 instances")
    print(f"Parallel execution: 8 threads")
    print("\nPress Ctrl+C to gracefully stop after current batch")
    print()
    
    # Define configurations
    configs_2s = [
        (10, 20, 2, 'results/ortools_gls_2s_production'),
        (20, 30, 2, 'results/ortools_gls_2s_production'),
        (50, 40, 2, 'results/ortools_gls_2s_production'),
        (100, 50, 2, 'results/ortools_gls_2s_production')
    ]
    
    configs_5s = [
        (10, 20, 5, 'results/ortools_gls_5s_production'),
        (20, 30, 5, 'results/ortools_gls_5s_production'),
        (50, 40, 5, 'results/ortools_gls_5s_production'),
        (100, 50, 5, 'results/ortools_gls_5s_production')
    ]
    
    all_configs = configs_2s + configs_5s
    
    print(f"Total configurations: {len(all_configs)}")
    for n, cap, timeout, _ in all_configs:
        print(f"  - N={n:3d}, Capacity={cap:2d}, Timeout={timeout}s")
    print()
    
    # Run configurations with parallel processing
    total_start = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_configuration, config): config 
                  for config in all_configs}
        
        results = {}
        for future in as_completed(futures):
            config = futures[future]
            try:
                completed = future.result()
                results[config] = completed
                n, _, timeout, _ = config
                print(f"\n[{datetime.now()}] Completed N={n}, timeout={timeout}s: {completed} instances")
            except Exception as e:
                print(f"\n[{datetime.now()}] Error for {config}: {e}")
                results[config] = 0
    
    # Summary
    total_time = time.time() - total_start
    print()
    print("=" * 70)
    print("Production Benchmark Summary")
    print("=" * 70)
    
    for config, completed in results.items():
        n, cap, timeout, _ = config
        status = "Complete" if completed >= 10000 else f"{completed}/10000"
        print(f"N={n:3d}, Timeout={timeout}s: {status}")
    
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now()}")
    
    # Create final summary
    summary_file = f'results/ortools_gls_production_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('results', exist_ok=True)
    
    summary = {
        'start_time': total_start,
        'end_time': time.time(),
        'total_hours': total_time/3600,
        'configurations': {}
    }
    
    for config, completed in results.items():
        n, cap, timeout, output_dir = config
        config_key = f"n{n}_timeout{timeout}"
        summary['configurations'][config_key] = {
            'n': n,
            'capacity': cap,
            'timeout': timeout,
            'completed_instances': completed,
            'target_instances': 10000,
            'output_dir': output_dir
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Check if all completed
    all_complete = all(c >= 10000 for c in results.values())
    return 0 if all_complete else 1

if __name__ == "__main__":
    sys.exit(main())
