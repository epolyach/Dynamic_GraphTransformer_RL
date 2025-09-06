#!/usr/bin/env python3
"""
Test runner for OR-Tools GLS benchmarks
Runs 20 instances per configuration for testing
"""

import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

def run_single_benchmark(config):
    """Run a single benchmark configuration"""
    n, capacity, timeout, output_dir = config
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{timestamp}] Starting: N={n}, timeout={timeout}s")
    
    # Use virtual environment Python
    venv_python = os.path.abspath('../venv/bin/python3')
    
    cmd = [
        venv_python, 'scripts/benchmark_ortools_gls_fixed.py',
        '--n', str(n),
        '--capacity', str(capacity),
        '--instances', '20',  # Test with 20 instances
        '--timeout', str(timeout)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'n{n}_cap{capacity}_{timeout}s_test.out')
        with open(output_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] ✓ Completed: N={n}, timeout={timeout}s")
            
            # Also save the JSON file to the results directory if it was created
            json_files = [f for f in os.listdir('.') if f.startswith(f'ortools_gls_n{n}_') and f.endswith('.json')]
            for jf in json_files:
                new_path = os.path.join(output_dir, jf)
                os.rename(jf, new_path)
                print(f"  Moved {jf} to {new_path}")
            
            return (True, config, output_file)
        else:
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] ✗ Failed: N={n}, timeout={timeout}s")
            return (False, config, result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] ⚠ Timeout: N={n}, timeout={timeout}s")
        return (False, config, "Process timeout")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] ✗ Error: N={n}, timeout={timeout}s - {str(e)}")
        return (False, config, str(e))

def main():
    print("=" * 70)
    print("OR-Tools GLS Test Benchmark Runner")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Configuration: 20 instances per test")
    print(f"Parallel threads: 8")
    print()
    
    # Define configurations: (n_customers, capacity, timeout, output_dir)
    configs_2s = [
        (10, 20, 2, 'results/ortools_gls_2s_test'),
        (20, 30, 2, 'results/ortools_gls_2s_test'),
        (50, 40, 2, 'results/ortools_gls_2s_test'),
        (100, 50, 2, 'results/ortools_gls_2s_test')
    ]
    
    configs_5s = [
        (10, 20, 5, 'results/ortools_gls_5s_test'),
        (20, 30, 5, 'results/ortools_gls_5s_test'),
        (50, 40, 5, 'results/ortools_gls_5s_test'),
        (100, 50, 5, 'results/ortools_gls_5s_test')
    ]
    
    all_configs = configs_2s + configs_5s
    
    print(f"Total configurations to run: {len(all_configs)}")
    print("Configurations:")
    for n, cap, timeout, _ in all_configs:
        print(f"  - N={n:3d}, Capacity={cap:2d}, Timeout={timeout}s")
    print()
    
    # Run benchmarks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_config = {executor.submit(run_single_benchmark, config): config 
                           for config in all_configs}
        
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception for config {config}: {e}")
                results.append((False, config, str(e)))
    
    # Summary
    print()
    print("=" * 70)
    print("Test Benchmark Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r[0])
    failed = len(results) - successful
    
    print(f"Total runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed configurations:")
        for success, config, error in results:
            if not success:
                n, cap, timeout, _ = config
                print(f"  - N={n}, timeout={timeout}s: {error[:100]}")
    
    print(f"\nEnd time: {datetime.now()}")
    
    # Create summary file
    summary_file = f'results/ortools_gls_test_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    os.makedirs('results', exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write(f"OR-Tools GLS Test Benchmark Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        for success, config, output in results:
            n, cap, timeout, _ = config
            status = "SUCCESS" if success else "FAILED"
            f.write(f"N={n:3d}, Cap={cap:2d}, Timeout={timeout}s: {status}\n")
            if success:
                f.write(f"  Output: {output}\n")
            else:
                f.write(f"  Error: {output[:200]}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
