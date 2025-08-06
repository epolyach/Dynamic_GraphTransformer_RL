#!/usr/bin/env python3
"""
Visualize CVRP Algorithm Comparison Results
Create comprehensive plots comparing Clarke-Wright and Nearest Neighbor algorithms
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

def create_comparison_plots():
    """Create comprehensive comparison plots"""
    
    # Data from our algorithm runs
    instances = list(range(1, 11))
    
    # Clarke-Wright results (None for failed instance 10)
    cw_distances = [1.31, 1.34, 1.13, 1.53, 1.42, 1.71, 1.40, 1.21, 1.20, None]
    cw_vehicles = [4, 4, 4, 4, 4, 4, 4, 4, 4, None]
    
    # Nearest Neighbor results
    nn_distances = [1.804, 1.746, 1.424, 1.860, 1.836, 1.967, 1.809, 1.825, 1.728, 1.803]
    nn_vehicles = [4, 4, 3, 4, 3, 4, 4, 4, 4, 5]
    
    # Instance demands for context
    demands = [11.1, 10.1, 8.4, 10.7, 7.4, 10.3, 9.5, 10.4, 10.7, 12.9]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Distance Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(instances))
    
    # Handle None values for Clarke-Wright
    cw_plot_distances = [d if d is not None else 0 for d in cw_distances]
    cw_colors = ['#2E86AB' if d is not None else '#FF6B6B' for d in cw_distances]
    
    bars1 = ax1.bar(x_pos - 0.2, cw_plot_distances, 0.4, 
                    label='Clarke-Wright', color=cw_colors, alpha=0.8)
    bars2 = ax1.bar(x_pos + 0.2, nn_distances, 0.4, 
                    label='Nearest Neighbor', color='#F18F01', alpha=0.8)
    
    # Add "FAILED" text for instance 10
    ax1.text(9-0.2, 0.05, 'FAILED', ha='center', va='bottom', 
             fontweight='bold', color='red', fontsize=10)
    
    ax1.set_xlabel('Instance ID', fontsize=12)
    ax1.set_ylabel('Total Distance', fontsize=12)
    ax1.set_title('Distance Comparison: Clarke-Wright vs Nearest Neighbor', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(instances)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Vehicle Usage Comparison
    ax2 = plt.subplot(2, 3, 2)
    cw_plot_vehicles = [v if v is not None else 0 for v in cw_vehicles]
    
    bars3 = ax2.bar(x_pos - 0.2, cw_plot_vehicles, 0.4, 
                    label='Clarke-Wright', color='#2E86AB', alpha=0.8)
    bars4 = ax2.bar(x_pos + 0.2, nn_vehicles, 0.4, 
                    label='Nearest Neighbor', color='#F18F01', alpha=0.8)
    
    ax2.text(9-0.2, 0.1, 'FAILED', ha='center', va='bottom', 
             fontweight='bold', color='red', fontsize=10)
    
    ax2.set_xlabel('Instance ID', fontsize=12)
    ax2.set_ylabel('Number of Vehicles Used', fontsize=12)
    ax2.set_title('Vehicle Usage Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(instances)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 6)
    
    # 3. Performance vs Demand Scatter Plot
    ax3 = plt.subplot(2, 3, 3)
    
    # Filter out None values for scatter plot
    cw_valid_distances = [d for d in cw_distances if d is not None]
    cw_valid_demands = [demands[i] for i, d in enumerate(cw_distances) if d is not None]
    
    ax3.scatter(cw_valid_demands, cw_valid_distances, 
               s=100, alpha=0.7, color='#2E86AB', label='Clarke-Wright', marker='o')
    ax3.scatter(demands, nn_distances, 
               s=100, alpha=0.7, color='#F18F01', label='Nearest Neighbor', marker='s')
    
    # Highlight failed instance
    ax3.scatter([demands[9]], [2.0], s=150, color='red', marker='x', 
               label='CW Failed', linewidth=3)
    
    ax3.set_xlabel('Total Demand', fontsize=12)
    ax3.set_ylabel('Total Distance', fontsize=12)
    ax3.set_title('Performance vs Problem Difficulty', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement Percentage
    ax4 = plt.subplot(2, 3, 4)
    improvements = []
    improvement_instances = []
    
    for i, (cw, nn) in enumerate(zip(cw_distances, nn_distances)):
        if cw is not None:
            improvement = ((nn - cw) / nn) * 100
            improvements.append(improvement)
            improvement_instances.append(i + 1)
    
    bars5 = ax4.bar(improvement_instances, improvements, 
                    color=['#28a745' if imp > 0 else '#dc3545' for imp in improvements], 
                    alpha=0.8)
    
    ax4.set_xlabel('Instance ID', fontsize=12)
    ax4.set_ylabel('Improvement (%)', fontsize=12)
    ax4.set_title('Clarke-Wright Improvement over Nearest Neighbor', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, imp in zip(bars5, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    # 5. Algorithm Success Rate
    ax5 = plt.subplot(2, 3, 5)
    algorithms = ['Clarke-Wright', 'Nearest Neighbor']
    success_rates = [90, 100]  # 9/10 vs 10/10
    colors = ['#2E86AB', '#F18F01']
    
    bars6 = ax5.bar(algorithms, success_rates, color=colors, alpha=0.8)
    ax5.set_ylabel('Success Rate (%)', fontsize=12)
    ax5.set_title('Algorithm Reliability', fontsize=14, fontweight='bold')
    ax5.set_ylim(0, 105)
    
    # Add percentage labels
    for bar, rate in zip(bars6, success_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    cw_valid = [d for d in cw_distances if d is not None]
    cw_avg = np.mean(cw_valid)
    nn_avg = np.mean(nn_distances)
    overall_improvement = np.mean(improvements)
    
    # Create summary text
    summary_text = f"""
ALGORITHM PERFORMANCE SUMMARY

Clarke-Wright Savings:
• Solved: 9/10 instances (90%)
• Avg Distance: {cw_avg:.3f}
• Best: {min(cw_valid):.3f}
• Worst: {max(cw_valid):.3f}
• Std Dev: {np.std(cw_valid):.3f}

Nearest Neighbor:
• Solved: 10/10 instances (100%)
• Avg Distance: {nn_avg:.3f}
• Best: {min(nn_distances):.3f}
• Worst: {max(nn_distances):.3f}
• Std Dev: {np.std(nn_distances):.3f}

COMPARISON:
• CW wins: 9/10 instances
• Avg improvement: {overall_improvement:.1f}%
• CW failed on highest demand (12.9)
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comprehensive comparison plot to plots/algorithm_comparison.png")
    
    return fig

def create_instance_detail_plots():
    """Create detailed plots for individual instances"""
    
    # Data
    instances = list(range(1, 11))
    cw_distances = [1.31, 1.34, 1.13, 1.53, 1.42, 1.71, 1.40, 1.21, 1.20, None]
    nn_distances = [1.804, 1.746, 1.424, 1.860, 1.836, 1.967, 1.809, 1.825, 1.728, 1.803]
    demands = [11.1, 10.1, 8.4, 10.7, 7.4, 10.3, 9.5, 10.4, 10.7, 12.9]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Distance comparison with trend lines
    solved_instances = [i for i, d in enumerate(cw_distances) if d is not None]
    solved_cw = [cw_distances[i] for i in solved_instances]
    solved_nn = [nn_distances[i] for i in solved_instances]
    solved_inst = [i+1 for i in solved_instances]
    
    ax1.plot(solved_inst, solved_cw, 'o-', color='#2E86AB', linewidth=2, 
             markersize=8, label='Clarke-Wright', alpha=0.8)
    ax1.plot(instances, nn_distances, 's-', color='#F18F01', linewidth=2, 
             markersize=8, label='Nearest Neighbor', alpha=0.8)
    
    # Mark failed instance
    ax1.plot([10], [max(nn_distances)], 'x', color='red', markersize=15, 
             markeredgewidth=3, label='CW Failed')
    
    ax1.set_xlabel('Instance ID', fontsize=12)
    ax1.set_ylabel('Total Distance', fontsize=12)
    ax1.set_title('Algorithm Performance Across All Instances', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(instances)
    
    # Plot 2: Demand difficulty analysis
    colors = ['green' if cw_distances[i] is not None else 'red' for i in range(10)]
    sizes = [50 + d*10 for d in demands]  # Scale bubble size by demand
    
    scatter = ax2.scatter(instances, demands, c=colors, s=sizes, alpha=0.6, 
                         edgecolors='black', linewidth=1)
    
    # Add demand threshold line
    ax2.axhline(y=12, color='red', linestyle='--', alpha=0.7, 
                label='Critical Demand Level')
    
    ax2.set_xlabel('Instance ID', fontsize=12)
    ax2.set_ylabel('Total Demand', fontsize=12)
    ax2.set_title('Problem Difficulty vs Algorithm Success', fontsize=14, fontweight='bold')
    ax2.legend(['Solved by CW', 'Failed by CW', 'Critical Level'])
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(instances)
    
    # Add annotations for interesting points
    ax2.annotate('Highest demand\n(CW failed)', xy=(10, 12.9), xytext=(8.5, 13.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center', color='red', fontweight='bold')
    
    ax2.annotate('Lowest demand\n(Best performance)', xy=(5, 7.4), xytext=(3, 6.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/instance_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved instance analysis plot to plots/instance_analysis.png")
    
    return fig

def main():
    """Main function to create all visualization plots"""
    print("🎨 Creating CVRP Algorithm Comparison Visualizations")
    print("=" * 60)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create comprehensive comparison plot
    print("\n📊 Creating comprehensive comparison plot...")
    fig1 = create_comparison_plots()
    
    # Create detailed instance analysis
    print("\n🔍 Creating detailed instance analysis...")
    fig2 = create_instance_detail_plots()
    
    print("\n🎉 All visualizations created successfully!")
    print("📁 Check the 'plots/' directory for:")
    print("   • algorithm_comparison.png - Comprehensive comparison")
    print("   • instance_analysis.png - Detailed instance analysis")
    
    # Display plots if running interactively
    plt.show()

if __name__ == '__main__':
    main()
