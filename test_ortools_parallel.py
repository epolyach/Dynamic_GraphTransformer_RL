#!/usr/bin/env python3
"""
Test script for OR-Tools GLS parallel runner
Small configuration for quick testing
"""

import os
import sys
import json
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

def run_thread_instances(args):
    """Run instances for a single thread"""
    n, capacity, timeout, thread_id, total_threads, num_instances, output_dir = args
    
    # Calculate which instances this thread handles (striped allocation)
    my_instances = list(range(thread_id, num_instances, total_threads))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}, timeout={timeout}s: processing {len(my_instances)} instances")
    
    # Check for virtual environment Python
    venv_python = 'venv/bin/python3' if os.path.exists('venv/bin/python3') else sys.executable
    
    cmd = [
        venv_python, 'benchmark_cpu/scripts/ortools/benchmarks/benchmark_ortools_gls_fixed.py',
        '--n', str(n),
        '--capacity', str(capacity),
        '--instances', str(len(my_instances)),
        '--timeout', str(timeout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=len(my_instances) * timeout + 30)
        
        if result.returncode == 0:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Look for generated JSON files
            json_files = [f for f in os.listdir('.') 
                         if f.startswith(f'ortools_gls_n{n}_') and f.endswith('.json')]
            
            if json_files:
                for jf in json_files:
                    new_name = f'thread_{thread_id:02d}_{jf}'
                    new_path = os.path.join(output_dir, new_name)
                    os.rename(jf, new_path)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}: saved to {new_path}")
                return True, new_path
            else:
                print(f"Warning: No JSON files found for N={n}, thread {thread_id}")
                print(f"stdout: {result.stdout[:200]}")
                print(f"stderr: {result.stderr[:200]}")
                return False, "No JSON produced"
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}: failed with code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
            return False, result.stderr[:200]
            
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}: timeout after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}: error - {str(e)}")
        return False, str(e)

def main():
    print("=" * 70)
    print("OR-Tools GLS Parallel Test Runner - SMALL TEST")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Small test configuration - just 2 instances per config with 2 threads
    configs = [
        (10, 20, 2, 2),   # N=10, capacity=20, timeout=2s, 2 threads (1 instance each)
        (20, 30, 2, 2),   # N=20, capacity=30, timeout=2s, 2 threads (1 instance each)
    ]
    
    # Create test output directory
    test_output_base = 'benchmark_gpu/results/ortools_test_runs'
    os.makedirs(test_output_base, exist_ok=True)
    
    # Prepare all tasks
    tasks = []
    for n, capacity, timeout, num_threads in configs:
        output_dir = f'{test_output_base}/n{n}_t{timeout}s_test'
        print(f"Config: N={n:3d}, Capacity={capacity}, Timeout={timeout:2d}s, Threads={num_threads}")
        for thread_id in range(num_threads):
            # Only 2 instances total, 1 per thread
            tasks.append((n, capacity, timeout, thread_id, num_threads, 2, output_dir))
    
    print(f"\nLaunching {len(tasks)} parallel processes...")
    print("This is a small test run - should complete in ~10 seconds")
    print()
    
    # Run all tasks in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_thread_instances, task) for task in tasks]
        
        # Wait for all to complete and collect results
        for i, future in enumerate(futures):
            try:
                success, output = future.result(timeout=30)
                results.append((tasks[i], success, output))
            except Exception as e:
                results.append((tasks[i], False, str(e)))
    
    elapsed = time.time() - start_time
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed:.2f} seconds")
    print()
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed tasks:")
        for task, success, output in results:
            if not success:
                n = task[0]
                thread_id = task[3]
                print(f"  N={n}, Thread={thread_id}: {output[:100]}")
    
    print()
    print("Test complete! Check the output directory:")
    print(f"  {test_output_base}/")
    
    return 0 if successful == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
