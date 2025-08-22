#!/usr/bin/env python3
"""
Visualize CVRP Solver Benchmark Results
Creates plots showing the performance comparison between DP and OR-Tools solvers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_benchmark_plots(csv_file: str = "cvrp_benchmark_results_corrected.csv"):
    """
    Create benchmark visualization plots from CSV results.
    
    Args:
        csv_file: Path to the CSV file with benchmark results
    """
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} data points")
        print(f"Problem sizes: N = {df['N'].min()} to {df['N'].max()}")
        print("\nData summary:")
        print(df.to_string(index=False))
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return
    
    # Create the plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CVRP Exact Solver Performance Comparison\n(DP vs OR-Tools)', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time vs Problem Size (Log Scale)
    ax1.semilogy(df['N'], df['time_dp'], 'o-', color='blue', linewidth=2, markersize=8, label='DP (Exact)', alpha=0.8)
    ax1.semilogy(df['N'], df['time_or'], 's-', color='red', linewidth=2, markersize=8, label='OR-Tools (Near-optimal)', alpha=0.8)
    ax1.set_xlabel('Number of Customers (N)', fontsize=12)
    ax1.set_ylabel('Average Time (seconds) - Log Scale', fontsize=12)
    ax1.set_title('Execution Time vs Problem Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xticks(df['N'])
    
    # Plot 2: Cost per Customer vs Problem Size
    ax2.plot(df['N'], df['cpc_dp'], 'o-', color='blue', linewidth=2, markersize=8, label='DP (Exact)', alpha=0.8)
    ax2.plot(df['N'], df['cpc_or'], 's-', color='red', linewidth=2, markersize=8, label='OR-Tools (Near-optimal)', alpha=0.8)
    
    # Add error bars for standard deviation
    ax2.errorbar(df['N'], df['cpc_dp'], yerr=df['std_dp'], fmt='o', color='blue', alpha=0.5, capsize=3)
    ax2.errorbar(df['N'], df['cpc_or'], yerr=df['std_or'], fmt='s', color='red', alpha=0.5, capsize=3)
    
    ax2.set_xlabel('Number of Customers (N)', fontsize=12)
    ax2.set_ylabel('Cost per Customer', fontsize=12)
    ax2.set_title('Solution Quality vs Problem Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(df['N'])
    
    # Plot 3: Time Ratio (OR-Tools / DP)
    time_ratio = df['time_or'] / df['time_dp']
    ax3.plot(df['N'], time_ratio, 'o-', color='green', linewidth=2, markersize=8, alpha=0.8)
    ax3.set_xlabel('Number of Customers (N)', fontsize=12)
    ax3.set_ylabel('Time Ratio (OR-Tools / DP)', fontsize=12)
    ax3.set_title('Speed Advantage: How much faster is DP?', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['N'])
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal performance')
    ax3.legend(fontsize=11)
    
    # Plot 4: Cost Gap (OR-Tools vs DP)
    cost_gap = ((df['cpc_or'] - df['cpc_dp']) / df['cpc_dp']) * 100
    ax4.plot(df['N'], cost_gap, 'o-', color='orange', linewidth=2, markersize=8, alpha=0.8)
    ax4.set_xlabel('Number of Customers (N)', fontsize=12)
    ax4.set_ylabel('Cost Gap (%)', fontsize=12)
    ax4.set_title('Quality Gap: How much better is DP?', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(df['N'])
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Equal cost')
    ax4.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cvrp_solver_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig('cvrp_solver_benchmark.pdf', bbox_inches='tight')
    print("\nPlots saved as 'cvrp_solver_benchmark.png' and 'cvrp_solver_benchmark.pdf'")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\nProblem Size Range: N = {df['N'].min()} to {df['N'].max()} customers")
    print(f"Fixed Parameters: Capacity = 30, Demand ∈ [1,10], Coordinates ∈ [0,1]²")
    print(f"Instances per N: 100")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Algorithm    | Avg Time (s) | Avg Cost/Customer | Quality  | Speed")
    print(f"-------------|--------------|-------------------|----------|--------")
    avg_time_dp = df['time_dp'].mean()
    avg_time_or = df['time_or'].mean()
    avg_cpc_dp = df['cpc_dp'].mean()
    avg_cpc_or = df['cpc_or'].mean()
    
    print(f"DP           | {avg_time_dp:11.4f} | {avg_cpc_dp:16.4f} | Exact    | {avg_time_or/avg_time_dp:5.1f}x")
    print(f"OR-Tools     | {avg_time_or:11.4f} | {avg_cpc_or:16.4f} | ~Optimal | 1.0x")
    
    print(f"\n🔍 KEY INSIGHTS:")
    max_ratio = time_ratio.max()
    min_ratio = time_ratio.min()
    avg_gap = cost_gap.mean()
    
    print(f"• DP is {min_ratio:.1f}x to {max_ratio:.1f}x FASTER than OR-Tools")
    print(f"• DP produces {avg_gap:.1f}% BETTER solutions on average (being exact)")
    print(f"• OR-Tools has higher computational overhead for small problems")
    print(f"• DP complexity is O(n²·2ⁿ), so it will become slower after ~N=15-20")
    
    print(f"\n⚠️  EXPECTED BEHAVIOR FOR LARGER N:")
    print(f"• DP will become exponentially slower as N increases")  
    print(f"• OR-Tools will maintain more consistent performance")
    print(f"• The crossover point is typically around N=15-20")

if __name__ == "__main__":
    create_benchmark_plots("cvrp_benchmark_results_corrected.csv")
