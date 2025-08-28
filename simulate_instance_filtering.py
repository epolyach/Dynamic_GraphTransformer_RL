#!/usr/bin/env python3
"""
Simulate the instance filtering that CPU benchmark does
This explains why CPU benchmark reports lower CPC values!
"""

import numpy as np
import sys
import time
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.exact_dp import solve as cpu_solve

def generate_and_solve_with_filtering(n_customers=6, n_target=100, timeout_per_instance=5.0):
    """Generate instances with CPU benchmark's filtering logic"""
    
    print(f"Simulating CPU benchmark with instance filtering")
    print(f"Target: {n_target} instances, timeout: {timeout_per_instance}s per instance")
    print("=" * 60)
    
    gen = EnhancedCVRPGenerator(config={})
    
    successful_cpcs = []
    all_cpcs = []  # Without filtering
    discarded_count = 0
    total_attempted = 0
    
    i = 0
    while len(successful_cpcs) < n_target:
        # Try up to 3 attempts per instance slot
        instance_found = False
        
        for attempt in range(3):
            total_attempted += 1
            seed = 4242 + n_customers * 1000 + i * 10 + attempt
            
            instance = gen.generate_instance(
                num_customers=n_customers,
                capacity=30,
                coord_range=100,
                demand_range=[1, 10],
                seed=seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            
            # Solve with timeout
            start = time.time()
            try:
                result = cpu_solve(instance, time_limit=timeout_per_instance, verbose=False)
                solve_time = time.time() - start
                cpc = result.cost / n_customers
                all_cpcs.append(cpc)  # Track all instances
                
                # Check if "too hard" (takes too long)
                if solve_time >= timeout_per_instance * 0.8:  # 80% of timeout threshold
                    discarded_count += 1
                    print(f"  Instance {i} attempt {attempt}: CPC={cpc:.4f}, Time={solve_time:.2f}s - DISCARDED (too hard)")
                    continue  # Try next attempt
                else:
                    successful_cpcs.append(cpc)
                    if len(successful_cpcs) <= 5:
                        print(f"  Instance {i}: CPC={cpc:.4f}, Time={solve_time:.2f}s - ACCEPTED")
                    instance_found = True
                    break
                    
            except Exception as e:
                discarded_count += 1
                print(f"  Instance {i} attempt {attempt}: Failed - {str(e)}")
                continue
        
        if not instance_found:
            print(f"  Instance {i}: All attempts failed/discarded")
        
        i += 1
        if i > n_target * 3:  # Safety limit
            break
    
    successful_cpcs = np.array(successful_cpcs[:n_target])
    all_cpcs = np.array(all_cpcs)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Total instances attempted: {total_attempted}")
    print(f"  Instances discarded: {discarded_count}")
    print(f"  Acceptance rate: {len(successful_cpcs)/total_attempted*100:.1f}%")
    
    print(f"\nWith filtering (easier instances only):")
    print(f"  Mean CPC: {successful_cpcs.mean():.6f}")
    print(f"  Std CPC:  {successful_cpcs.std():.6f}")
    
    print(f"\nWithout filtering (all instances):")
    print(f"  Mean CPC: {all_cpcs.mean():.6f}")
    print(f"  Std CPC:  {all_cpcs.std():.6f}")
    
    print(f"\nDifference due to filtering: {all_cpcs.mean() - successful_cpcs.mean():.6f}")
    
    print("\nComparison with reported values:")
    print("  CPU benchmark (with filtering): Mean CPC = 0.465060")
    print("  GPU benchmark (no filtering):   Mean CPC = 0.478376")
    print(f"  Our filtered simulation:        Mean CPC = {successful_cpcs.mean():.6f}")
    print(f"  Our unfiltered results:         Mean CPC = {all_cpcs.mean():.6f}")

if __name__ == "__main__":
    # Use more aggressive timeout to see filtering effect
    generate_and_solve_with_filtering(n_customers=6, n_target=100, timeout_per_instance=0.3)
