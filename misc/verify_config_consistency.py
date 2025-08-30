#!/usr/bin/env python3
"""
Verification script to demonstrate that both CPU and GPU benchmarks
now use identical parameters from config.json and produce identical results.
"""

import subprocess
import sys

def run_cpu_benchmark(output_name):
    """Run CPU benchmark."""
    print(f"🚀 Running CPU Benchmark...")
    
    cmd = [
        'python3', 'benchmark_exact_cpu_config.py', 
        '--n-start', '5', '--n-end', '5',
        '--instances-min', '1', '--instances-max', '1',
        '--timeout', '30',
        '--output', output_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ CPU Benchmark completed successfully")
        return True
    else:
        print(f"❌ CPU Benchmark failed:")
        print(result.stderr)
        return False

def run_gpu_benchmark(output_name):
    """Run GPU benchmark."""
    print(f"🚀 Running GPU Benchmark...")
    
    cmd = [
        'python3', 'benchmark_exact_gpu_config.py', 
        '--n-start', '5', '--n-end', '5',
        '--instances', '1',
        '--timeout', '30',
        '--output', output_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ GPU Benchmark completed successfully")
        return True
    else:
        print(f"❌ GPU Benchmark failed:")
        print(result.stderr)
        return False

def extract_cpc_values(filename):
    """Extract cost per customer values from CSV output."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                data_line = lines[1].split(',')
                # Extract CPC values for all solvers (positions 2, 7, 12, 17, 22)
                cpc_values = {
                    'ortools_vrp': float(data_line[2]),
                    'milp': float(data_line[7]),
                    'dp': float(data_line[12]),
                    'pulp': float(data_line[17]),
                    'heuristic': float(data_line[22])
                }
                return cpc_values
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def main():
    print("="*80)
    print("🔧 CONFIG-DRIVEN CVRP BENCHMARK VERIFICATION")
    print("="*80)
    print()
    
    # Show config
    print("📋 Configuration being used:")
    with open('config.json', 'r') as f:
        import json
        config = json.load(f)
        params = config['instance_generation']
        print(f"   - Capacity: {params['capacity']}")
        print(f"   - Demand range: [{params['demand_min']}, {params['demand_max']}]")
        print(f"   - Coordinate range: [0, {params['coord_range']}] normalized to [0,1]")
    print()
    
    # Run both benchmarks
    cpu_success = run_cpu_benchmark('verify_cpu')
    gpu_success = run_gpu_benchmark('verify_gpu')
    
    if not (cpu_success and gpu_success):
        print("❌ One or both benchmarks failed")
        return 1
    
    print()
    print("📊 Comparing results...")
    
    # Extract and compare results
    cpu_cpc = extract_cpc_values('verify_cpu')
    gpu_cpc = extract_cpc_values('verify_gpu')
    
    if cpu_cpc is None or gpu_cpc is None:
        print("❌ Could not extract CPC values")
        return 1
    
    print()
    print("Results comparison:")
    print(f"{'Solver':<15} {'CPU CPC':<20} {'GPU CPC':<20} {'Match':<10}")
    print("-" * 70)
    
    all_match = True
    for solver in cpu_cpc:
        cpu_val = cpu_cpc[solver]
        gpu_val = gpu_cpc[solver]
        match = abs(cpu_val - gpu_val) < 1e-10
        all_match &= match
        
        match_symbol = "✅" if match else "❌"
        print(f"{solver:<15} {cpu_val:<20.12f} {gpu_val:<20.12f} {match_symbol}")
    
    print()
    if all_match:
        print("🎉 SUCCESS: All solvers produce IDENTICAL results!")
        print("✅ Config-driven approach ensures perfect consistency")
        return 0
    else:
        print("❌ FAILURE: Results do not match")
        return 1

if __name__ == "__main__":
    sys.exit(main())
