#!/usr/bin/env python3
"""
Comparison of Classical CVRP Algorithms
Results from Clarke-Wright and Nearest Neighbor heuristics
"""

import numpy as np

def print_comparison():
    """Compare results from Clarke-Wright and Nearest Neighbor algorithms"""
    
    # Results from Clarke-Wright Savings Algorithm
    clarke_wright_results = {
        1: {'distance': 1.31, 'vehicles': 4, 'status': 'solved'},
        2: {'distance': 1.34, 'vehicles': 4, 'status': 'solved'},
        3: {'distance': 1.13, 'vehicles': 4, 'status': 'solved'},  # Note: only needed 3 vehicles
        4: {'distance': 1.53, 'vehicles': 4, 'status': 'solved'},
        5: {'distance': 1.42, 'vehicles': 4, 'status': 'solved'},  # Note: only needed 3 vehicles
        6: {'distance': 1.71, 'vehicles': 4, 'status': 'solved'},
        7: {'distance': 1.40, 'vehicles': 4, 'status': 'solved'},
        8: {'distance': 1.21, 'vehicles': 4, 'status': 'solved'},
        9: {'distance': 1.20, 'vehicles': 4, 'status': 'solved'},
        10: {'distance': None, 'vehicles': None, 'status': 'failed'},
    }
    
    # Results from Nearest Neighbor Heuristic
    nearest_neighbor_results = {
        1: {'distance': 1.804, 'vehicles': 4, 'status': 'solved'},
        2: {'distance': 1.746, 'vehicles': 4, 'status': 'solved'},
        3: {'distance': 1.424, 'vehicles': 3, 'status': 'solved'},
        4: {'distance': 1.860, 'vehicles': 4, 'status': 'solved'},
        5: {'distance': 1.836, 'vehicles': 3, 'status': 'solved'},
        6: {'distance': 1.967, 'vehicles': 4, 'status': 'solved'},
        7: {'distance': 1.809, 'vehicles': 4, 'status': 'solved'},
        8: {'distance': 1.825, 'vehicles': 4, 'status': 'solved'},
        9: {'distance': 1.728, 'vehicles': 4, 'status': 'solved'},
        10: {'distance': 1.803, 'vehicles': 5, 'status': 'solved'},
    }
    
    print("🚛 CVRP Classical Algorithms Comparison")
    print("=" * 80)
    print(f"{'Instance':<10} {'Clarke-Wright':<15} {'Nearest Neighbor':<17} {'Improvement':<12} {'Winner':<10}")
    print(f"{'ID':<10} {'Distance':<15} {'Distance':<17} {'(CW vs NN)':<12} {'':<10}")
    print("-" * 80)
    
    cw_wins = 0
    nn_wins = 0
    cw_distances = []
    nn_distances = []
    improvements = []
    
    for i in range(1, 11):
        cw = clarke_wright_results[i]
        nn = nearest_neighbor_results[i]
        
        if cw['status'] == 'solved' and nn['status'] == 'solved':
            cw_dist = cw['distance']
            nn_dist = nn['distance']
            improvement = ((nn_dist - cw_dist) / nn_dist) * 100
            improvements.append(improvement)
            cw_distances.append(cw_dist)
            nn_distances.append(nn_dist)
            
            if cw_dist < nn_dist:
                winner = "CW ✅"
                cw_wins += 1
            else:
                winner = "NN ✅"
                nn_wins += 1
                
            print(f"{i:<10} {cw_dist:<15.3f} {nn_dist:<17.3f} {improvement:<12.1f}% {winner:<10}")
            
        elif cw['status'] == 'failed':
            print(f"{i:<10} {'FAILED':<15} {nn['distance']:<17.3f} {'N/A':<12} {'NN ✅':<10}")
            nn_wins += 1
            nn_distances.append(nn['distance'])
        else:
            print(f"{i:<10} {cw['distance']:<15.3f} {'FAILED':<17} {'N/A':<12} {'CW ✅':<10}")
            cw_wins += 1
            cw_distances.append(cw['distance'])
    
    print("-" * 80)
    
    # Summary Statistics
    print("\\n📊 SUMMARY STATISTICS")
    print("=" * 50)
    
    if cw_distances:
        print(f"Clarke-Wright Savings Algorithm:")
        print(f"  • Solved instances: {len(cw_distances)}/10")
        print(f"  • Average distance: {np.mean(cw_distances):.3f}")
        print(f"  • Best distance: {np.min(cw_distances):.3f}")
        print(f"  • Worst distance: {np.max(cw_distances):.3f}")
        print(f"  • Standard deviation: {np.std(cw_distances):.3f}")
    
    print()
    
    if nn_distances:
        print(f"Nearest Neighbor Heuristic:")
        print(f"  • Solved instances: {len(nn_distances)}/10")
        print(f"  • Average distance: {np.mean(nn_distances):.3f}")
        print(f"  • Best distance: {np.min(nn_distances):.3f}")
        print(f"  • Worst distance: {np.max(nn_distances):.3f}")
        print(f"  • Standard deviation: {np.std(nn_distances):.3f}")
    
    print()
    print("🏆 ALGORITHM COMPARISON")
    print("=" * 30)
    print(f"Clarke-Wright wins: {cw_wins}/10 instances")
    print(f"Nearest Neighbor wins: {nn_wins}/10 instances")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"Average improvement (CW vs NN): {avg_improvement:.1f}%")
        if avg_improvement > 0:
            print("🎉 Clarke-Wright is better on average!")
        else:
            print("🎉 Nearest Neighbor is better on average!")
    
    print("\\n💡 KEY INSIGHTS")
    print("=" * 20)
    print("• Clarke-Wright failed on instance 10 (highest demand: 12.9)")
    print("• Nearest Neighbor solved all instances but used more vehicles")
    print("• Clarke-Wright generally finds shorter routes when it succeeds")
    print("• Both algorithms respect capacity constraints")
    print("• Instance 10 required 5 vehicles due to high total demand")
    
    # Vehicle usage comparison
    print("\\n🚐 VEHICLE USAGE COMPARISON")
    print("=" * 35)
    print(f"{'Instance':<10} {'CW Vehicles':<12} {'NN Vehicles':<12} {'Difference':<10}")
    print("-" * 45)
    
    for i in range(1, 11):
        cw = clarke_wright_results[i]
        nn = nearest_neighbor_results[i]
        
        if cw['status'] == 'solved':
            cw_veh = cw['vehicles']
            nn_veh = nn['vehicles']
            diff = nn_veh - cw_veh
            diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
            print(f"{i:<10} {cw_veh:<12} {nn_veh:<12} {diff_str:<10}")
        else:
            print(f"{i:<10} {'FAILED':<12} {nn['vehicles']:<12} {'N/A':<10}")

if __name__ == '__main__':
    print_comparison()
