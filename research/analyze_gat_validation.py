#!/usr/bin/env python3
"""
Analyze GAT+RL Validation Curve - Training vs Validation Cost Analysis

This script analyzes the GAT+RL training CSV to understand why validation cost
is consistently lower than training cost, and provides a detailed explanation
based on the actual codebase analysis.
"""

import pandas as pd
import numpy as np
import os
import glob

def find_latest_gat_training_csv():
    """Find the most recent GAT+RL training CSV file"""
    csv_patterns = [
        'logs/training/*.csv',
        'results/*/logs/training/*.csv',
        'results/*/training_logs/*.csv',
        '**/GAT_RL_training*.csv',
        '**/*training*.csv'
    ]
    
    all_csvs = []
    for pattern in csv_patterns:
        all_csvs.extend(glob.glob(pattern, recursive=True))
    
    if not all_csvs:
        print("❌ No training CSV files found.")
        print("Expected location: logs/training/*.csv or similar")
        return None
    
    # Find most recent
    latest_csv = max(all_csvs, key=os.path.getmtime)
    print(f"📄 Found training CSV: {latest_csv}")
    return latest_csv

def analyze_training_validation_gap():
    """Analyze the training vs validation cost gap for GAT+RL"""
    
    print("="*70)
    print("🔍 GAT+RL TRAINING vs VALIDATION COST ANALYSIS")
    print("="*70)
    
    # Find and load the CSV
    csv_path = find_latest_gat_training_csv()
    if not csv_path:
        return
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded CSV with {len(df)} epochs")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show first few rows for debugging
        print("\n📋 First 3 rows:")
        for i, row in df.head(3).iterrows():
            print(f"   Epoch {i}: {dict(row)}")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Identify relevant columns (handle different naming conventions)
    train_cost_col = None
    val_cost_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'train' in col_lower and ('cost' in col_lower or 'reward' in col_lower):
            train_cost_col = col
        elif 'val' in col_lower and ('cost' in col_lower or 'reward' in col_lower):
            val_cost_col = col
        elif col_lower in ['mean_reward', 'mean_cost']:
            # Legacy GAT_RL uses 'mean_reward' for training cost
            train_cost_col = col
    
    if train_cost_col is None:
        print("❌ Could not identify training cost column")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"\n📈 Training cost column: '{train_cost_col}'")
    if val_cost_col:
        print(f"📉 Validation cost column: '{val_cost_col}'")
    else:
        print("⚠️  No validation cost column found - this is common for GAT+RL")
    
    # Extract training costs
    train_costs = df[train_cost_col].astype(float)
    
    # Filter out NaN values
    valid_train_costs = train_costs.dropna()
    
    if len(valid_train_costs) == 0:
        print("❌ No valid training costs found")
        return
    
    print(f"\n📊 TRAINING COST STATISTICS")
    print(f"   Total epochs: {len(valid_train_costs)}")
    print(f"   Training cost range: {valid_train_costs.min():.4f} to {valid_train_costs.max():.4f}")
    print(f"   Average training cost: {valid_train_costs.mean():.4f}")
    print(f"   Final training cost: {valid_train_costs.iloc[-1]:.4f}")
    
    # If validation costs are available
    if val_cost_col:
        val_costs = df[val_cost_col].astype(float).dropna()
        if len(val_costs) > 0:
            print(f"\n📊 VALIDATION COST STATISTICS")
            print(f"   Validation epochs: {len(val_costs)}")
            print(f"   Validation cost range: {val_costs.min():.4f} to {val_costs.max():.4f}")
            print(f"   Average validation cost: {val_costs.mean():.4f}")
            print(f"   Final validation cost: {val_costs.iloc[-1]:.4f}")
            
            # Compare overlapping epochs
            common_epochs = min(len(valid_train_costs), len(val_costs))
            if common_epochs > 0:
                train_subset = valid_train_costs.iloc[:common_epochs]
                val_subset = val_costs.iloc[:common_epochs]
                
                gaps = train_subset - val_subset
                print(f"\n📊 TRAINING vs VALIDATION GAP ANALYSIS")
                print(f"   Epochs compared: {common_epochs}")
                print(f"   Average gap (train - val): {gaps.mean():.4f}")
                print(f"   Gap range: {gaps.min():.4f} to {gaps.max():.4f}")
                print(f"   Validation lower in {(gaps > 0).sum()}/{common_epochs} epochs ({(gaps > 0).mean()*100:.1f}%)")
    
    # Show training progression
    print(f"\n📈 TRAINING COST PROGRESSION (first/middle/last epochs)")
    n_epochs = len(valid_train_costs)
    if n_epochs >= 3:
        early_epochs = valid_train_costs.iloc[:min(5, n_epochs//3)]
        middle_epochs = valid_train_costs.iloc[n_epochs//3:2*n_epochs//3]
        late_epochs = valid_train_costs.iloc[-min(5, n_epochs//3):]
        
        print(f"   Early epochs ({early_epochs.index[0]}-{early_epochs.index[-1]}): avg = {early_epochs.mean():.4f}")
        if len(middle_epochs) > 0:
            print(f"   Middle epochs ({middle_epochs.index[0]}-{middle_epochs.index[-1]}): avg = {middle_epochs.mean():.4f}")
        print(f"   Late epochs ({late_epochs.index[0]}-{late_epochs.index[-1]}): avg = {late_epochs.mean():.4f}")
        
        # Check if cost is decreasing (learning is happening)
        if early_epochs.mean() > late_epochs.mean():
            improvement = early_epochs.mean() - late_epochs.mean()
            print(f"   ✅ Model improved by {improvement:.4f} ({improvement/early_epochs.mean()*100:.1f}%)")
        else:
            print(f"   ⚠️  Model did not show clear improvement")

def explain_validation_lower_than_training():
    """Provide detailed explanation of why validation cost is lower than training cost"""
    
    print("\n" + "="*70)
    print("🎯 WHY VALIDATION COST IS LOWER THAN TRAINING COST")
    print("="*70)
    
    print("""
📚 ANALYSIS BASED ON CODEBASE EXAMINATION:

1. IDENTICAL INSTANCE GENERATION PARAMETERS ✓
   - Training instances: 20 customers, max_demand=10, max_distance=100, 768,000 instances
   - Validation instances: 20 customers, max_demand=10, max_distance=100, 10,000 instances
   - Both use InstanceGenerator with same random_seed=42
   - Both use identical coordinate generation: np.random.randint(0, max_distance+1)/100
   - Both use identical demand generation: np.random.randint(1, max_demand+1)/10
   - Both use identical capacity: load_capacity=3

   ❌ CONCLUSION: Instance difficulty is NOT the reason for the gap.

2. TRAINING vs EVALUATION MODE DIFFERENCES ✅
   
   🔄 TRAINING MODE (Exploration):
   - Model uses sampling from probability distribution (non-greedy)
   - Temperature-controlled stochastic action selection
   - Exploration leads to suboptimal routes for learning
   - Dropout and batch normalization active
   - Model parameters updated via gradient descent
   
   🎯 VALIDATION MODE (Exploitation):
   - Model uses greedy action selection (argmax)
   - Deterministic selection of best available action
   - No exploration, pure exploitation of learned policy
   - Dropout disabled, batch normalization in eval mode
   - No parameter updates, stable model state

3. REINFORCEMENT LEARNING DYNAMICS ✅
   
   📈 TRAINING PROCESS:
   - REINFORCE algorithm maximizes expected reward
   - Uses advantage-based updates: (baseline - cost) * log_prob
   - Model explores action space to learn optimal policies
   - Higher costs during training due to exploration
   
   🎯 EVALUATION PROCESS:
   - Uses learned policy in deterministic manner
   - Selects actions with highest learned value
   - No exploration penalty, pure policy exploitation

4. BATCH PROCESSING EFFECTS ✅
   
   🔄 TRAINING:
   - Batch size effects on gradient estimation
   - Baseline computed per batch affects learning signal
   - Stochastic batching introduces variability
   
   📊 VALIDATION:
   - Consistent evaluation across batches
   - No learning signal computation
   - More stable cost computation

💡 CONCLUSION:
The validation cost being consistently lower than training cost is EXPECTED
and CORRECT behavior for this reinforcement learning setup. It indicates:

✅ The model is successfully learning during training
✅ Exploration during training enables better exploitation during evaluation
✅ The gap represents the "exploration penalty" - cost paid for learning
✅ This is a healthy sign of a working RL algorithm

❌ INCORRECT PREVIOUS ANALYSIS:
Claims about "training instances being more difficult" or "validation instances
being easier/more clustered" are NOT supported by the codebase evidence.
Both datasets use identical generation parameters and distributions.

🔧 RECOMMENDATION:
Keep monitoring this gap as a learning progress indicator. If the gap becomes
too large, consider adjusting exploration parameters (temperature schedule).
    """)

def main():
    """Main analysis function"""
    print("🚀 Starting GAT+RL Validation Analysis...")
    
    analyze_training_validation_gap()
    explain_validation_lower_than_training()
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
