import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the optimized GAT training data
gat_optimized = pd.read_csv('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny_gat/csv/history_gat_rl.csv')

# Read other models' data for comparison (from tiny config)
gt = pd.read_csv('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny/csv/history_gt_rl.csv')
dgt = pd.read_csv('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny/csv/history_dgt_rl.csv')

# Also read the original GAT for comparison if it exists
try:
    gat_original = pd.read_csv('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny/csv/history_gat_rl.csv')
    has_original = True
except:
    has_original = False

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Policy Loss Comparison (all models)
ax = axes[0, 0]
ax.plot(gt['epoch'], gt['train_loss'], label='GT+RL', alpha=0.8, linewidth=2)
ax.plot(dgt['epoch'], dgt['train_loss'], label='DGT+RL', alpha=0.8, linewidth=2)
ax.plot(gat_optimized['epoch'], gat_optimized['train_loss'], label='GAT+RL (Optimized)', alpha=0.8, linewidth=2, color='green')
if has_original:
    ax.plot(gat_original['epoch'], gat_original['train_loss'], label='GAT+RL (Original)', alpha=0.5, linewidth=1, linestyle='--', color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Policy Loss')
ax.set_title('Policy Loss Comparison - All Models')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. GAT Loss Improvement (before vs after optimization)
ax = axes[0, 1]
ax.plot(gat_optimized['epoch'], gat_optimized['train_loss'], label='GAT Optimized Config', alpha=0.8, linewidth=2, color='green')
if has_original:
    ax.plot(gat_original['epoch'], gat_original['train_loss'], label='GAT Original Config', alpha=0.8, linewidth=2, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Policy Loss')
ax.set_title('GAT Loss: Original vs Optimized Config')
ax.legend()
ax.grid(True, alpha=0.3)
# Add annotation about scale improvement
if has_original:
    original_scale = np.mean(np.abs(gat_original['train_loss'].dropna()))
    optimized_scale = np.mean(np.abs(gat_optimized['train_loss'].dropna()))
    improvement = optimized_scale / original_scale
    ax.text(0.05, 0.95, f'Scale improvement: {improvement:.1f}x', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Validation Cost Comparison
ax = axes[1, 0]
# Filter out NaN values for validation cost
gt_val = gt[['epoch', 'val_cost']].dropna()
dgt_val = dgt[['epoch', 'val_cost']].dropna()
gat_optimized_val = gat_optimized[['epoch', 'val_cost']].dropna()

ax.plot(gt_val['epoch'], gt_val['val_cost'], 'o-', label='GT+RL', alpha=0.8, markersize=4)
ax.plot(dgt_val['epoch'], dgt_val['val_cost'], 's-', label='DGT+RL', alpha=0.8, markersize=4)
ax.plot(gat_optimized_val['epoch'], gat_optimized_val['val_cost'], '^-', label='GAT+RL (Optimized)', alpha=0.8, markersize=4, color='green')
if has_original:
    gat_original_val = gat_original[['epoch', 'val_cost']].dropna()
    ax.plot(gat_original_val['epoch'], gat_original_val['val_cost'], 'v-', label='GAT+RL (Original)', alpha=0.5, markersize=3, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Cost')
ax.set_title('Validation Cost Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Learning Rate and Temperature Schedule (GAT Optimized)
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(gat_optimized['epoch'], gat_optimized['learning_rate'], 'b-', label='Learning Rate', alpha=0.8)
ax2.plot(gat_optimized['epoch'], gat_optimized['temperature'], 'r-', label='Temperature', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate', color='b')
ax2.set_ylabel('Temperature', color='r')
ax.set_title('GAT Optimized: LR & Temperature Schedule')
ax.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax.grid(True, alpha=0.3)
# Add legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.suptitle('GAT Model Training Analysis: Optimized Configuration', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny_gat/gat_optimized_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("GAT OPTIMIZATION SUMMARY")
print("="*60)

print("\n1. Loss Scale Analysis:")
if has_original:
    # Focus on the initial loss which shows the entropy contribution effect
    original_initial = np.abs(gat_original['train_loss'].iloc[0])
    optimized_initial = np.abs(gat_optimized['train_loss'].iloc[0])
    
    # Also check max values (which show peak learning)
    original_max = np.max(np.abs(gat_original['train_loss'].dropna()))
    optimized_max = np.max(np.abs(gat_optimized['train_loss'].dropna()))
    
    print(f"   Initial loss magnitude:")
    print(f"     Original:  {original_initial:.2e}")
    print(f"     Optimized: {optimized_initial:.2e}")
    print(f"     Ratio: {optimized_initial/original_initial:.2f}x")
    
    print(f"   Peak loss magnitude:")
    print(f"     Original:  {original_max:.2e}")
    print(f"     Optimized: {optimized_max:.2e}")
    
    print(f"\n   Note: Higher entropy coefficient (0.1 vs 0.01) initially increases loss visibility")
    print(f"         but faster learning rate leads to quicker convergence")
else:
    optimized_initial = np.abs(gat_optimized['train_loss'].iloc[0])
    optimized_max = np.max(np.abs(gat_optimized['train_loss'].dropna()))
    print(f"   Initial loss: {optimized_initial:.2e}")
    print(f"   Peak loss: {optimized_max:.2e}")

print("\n2. Final Performance:")
print(f"   Final validation cost: {gat_optimized['val_cost'].dropna().iloc[-1]:.4f}")
print(f"   Best validation cost: {gat_optimized['val_cost'].dropna().min():.4f}")
print(f"   Best epoch: {gat_optimized.loc[gat_optimized['val_cost'].idxmin(), 'epoch']:.0f}")

print("\n3. Hyperparameter Changes:")
print("   Learning rate: 0.0003 -> 0.001 (3.3x increase)")
print("   Entropy coefficient: 0.01 -> 0.1 (10x increase)")  
print("   Temperature start: 5.0 -> 2.0 (gentler)")
print("   Temperature min: 0.2 -> 0.5 (higher floor)")
print("   Gradient clipping: 2.0 -> 1.0 (tighter)")

print("\n4. Training Dynamics:")
print(f"   Total epochs: {len(gat_optimized)}")
print(f"   Convergence behavior: {'Stable' if gat_optimized['train_loss'].rolling(10).std().dropna().mean() < 1e-4 else 'Variable'}")

# Compare with other models
print("\n5. Model Comparison (Best Validation Cost):")
print(f"   GT+RL:  {gt['val_cost'].dropna().min():.4f}")
print(f"   DGT+RL: {dgt['val_cost'].dropna().min():.4f}")
print(f"   GAT+RL (Optimized): {gat_optimized['val_cost'].dropna().min():.4f}")
if has_original:
    print(f"   GAT+RL (Original): {gat_original['val_cost'].dropna().min():.4f}")

print("="*60)
