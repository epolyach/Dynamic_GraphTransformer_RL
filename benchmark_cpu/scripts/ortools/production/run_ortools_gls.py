#!/usr/bin/env python3
"""
Parallel test runner for OR-Tools GLS benchmarks.
Each thread processes its share of instances and produces ONE JSON file.
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
    
    # Use virtual environment Python
    venv_python = os.path.abspath('../venv/bin/python3')
    
    cmd = [
        venv_python, 'scripts/benchmark_ortools_gls_fixed.py',
        '--n', str(n),
        '--capacity', str(capacity),
        '--instances', str(len(my_instances)),
        '--timeout', str(timeout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=len(my_instances) * timeout + 60)
        
        if result.returncode == 0:
            # Move the JSON file to output directory with thread ID
            os.makedirs(output_dir, exist_ok=True)
            json_files = [f for f in os.listdir('.') 
                         if f.startswith(f'ortools_gls_n{n}_') and f.endswith('.json')]
            
            for jf in json_files:
                new_name = f'thread_{thread_id:02d}_{jf}'
                new_path = os.path.join(output_dir, new_name)
                os.rename(jf, new_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}, timeout={timeout}s: completed ✓")
                return True, new_path
            return False, "No JSON produced"
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}, timeout={timeout}s: failed ✗")
            return False, result.stderr[:200]
            
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}, timeout={timeout}s: timeout ⚠")
        return False, "Timeout"
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id} for N={n}, timeout={timeout}s: error - {str(e)}")
        return False, str(e)

def main():
    print("=" * 70)
    print("OR-Tools GLS Parallel Test Runner")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print("Target: 20 instances per configuration")
    print("Total threads: 18 (2+4+2+4+6)")
    print()
    
    # Define configurations with their thread counts
    configs = [
        (50, 40, 10, 2),   # N=50, capacity=40, timeout=10s, 2 threads
        (50, 40, 20, 4),   # N=50, capacity=40, timeout=20s, 4 threads
        (100, 50, 10, 2),  # N=100, capacity=50, timeout=10s, 2 threads
        (100, 50, 20, 4),  # N=100, capacity=50, timeout=20s, 4 threads
        (100, 50, 30, 6),  # N=100, capacity=50, timeout=30s, 6 threads
    ]
    
    # Prepare all tasks
    tasks = []
    for n, capacity, timeout, num_threads in configs:
        output_dir = f'results/ortools_gls_{timeout}s_test'
        print(f"Config: N={n:3d}, Capacity={capacity}, Timeout={timeout:2d}s, Threads={num_threads}")
        for thread_id in range(num_threads):
            tasks.append((n, capacity, timeout, thread_id, num_threads, 20, output_dir))
    
    print(f"\nLaunching {len(tasks)} parallel processes...")
    print("Expected completion time: ~120 seconds")
    print()
    
    # Run all tasks in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=18) as executor:
        futures = [executor.submit(run_thread_instances, task) for task in tasks]
        
        # Wait for all to complete
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception: {e}")
                results.append((False, str(e)))
    
    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r[0])
    failed = len(results) - successful
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total threads: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    
    # Check JSON files per directory
    print("\nJSON files per configuration:")
    for n, capacity, timeout, num_threads in configs:
        output_dir = f'results/ortools_gls_{timeout}s_test'
        if os.path.exists(output_dir):
            json_count = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
            print(f"  N={n:3d}, timeout={timeout:2d}s: {json_count} JSON files (expected {num_threads})")
    
    print(f"\nEnd time: {datetime.now()}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
