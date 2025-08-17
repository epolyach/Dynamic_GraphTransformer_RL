#!/usr/bin/env python3
"""
Quick verification script to check if training data is being loaded properly for all models.
"""

import pandas as pd
import os

def verify_training_data():
    """Verify training data loading for all models"""
    csv_dir = "results/small/csv"
    
    # Map display names to CSV keys (same as in plotting script)
    name_to_key = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'DGT-Ultra+RL': 'dgt_ultra_rl',
        'DGT-Lite+RL': 'dgt_lite_rl',
        'DGT-Super+RL': 'dgt_super_rl',
        'GT-Lite+RL': 'gt_lite_rl',
        'GT-Ultra+RL': 'gt_ultra_rl',
    }
    
    print("📊 TRAINING DATA VERIFICATION")
    print("=" * 80)
    
    for model_name, key in name_to_key.items():
        filepath = os.path.join(csv_dir, f"history_{key}.csv")
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Count data points
                total_epochs = len(df)
                train_loss_count = df['train_loss'].notna().sum()
                train_cost_count = df['train_cost'].notna().sum()
                val_cost_count = df['val_cost'].notna().sum()
                
                # Get final values
                final_train_loss = df['train_loss'].dropna().iloc[-1] if train_loss_count > 0 else "N/A"
                final_train_cost = df['train_cost'].dropna().iloc[-1] if train_cost_count > 0 else "N/A"
                final_val_cost = df['val_cost'].dropna().iloc[-1] if val_cost_count > 0 else "N/A"
                
                print(f"\n✅ {model_name}:")
                print(f"   📁 File: history_{key}.csv")
                print(f"   📊 Total epochs: {total_epochs}")
                print(f"   🔥 Train loss points: {train_loss_count} (final: {final_train_loss:.4f})" if isinstance(final_train_loss, float) else f"   🔥 Train loss points: {train_loss_count} (final: {final_train_loss})")
                print(f"   💰 Train cost points: {train_cost_count} (final: {final_train_cost:.4f})" if isinstance(final_train_cost, float) else f"   💰 Train cost points: {train_cost_count} (final: {final_train_cost})")
                print(f"   ✨ Val cost points: {val_cost_count} (final: {final_val_cost:.4f})" if isinstance(final_val_cost, float) else f"   ✨ Val cost points: {val_cost_count} (final: {final_val_cost})")
                
                # Check for missing columns
                required_cols = ['epoch', 'train_loss', 'train_cost', 'val_cost']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"   ⚠️ Missing columns: {missing_cols}")
                
            except Exception as e:
                print(f"\n❌ {model_name}: Error reading CSV - {e}")
        else:
            print(f"\n❌ {model_name}: CSV file not found at {filepath}")
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    
    # Count available models
    available = 0
    with_training = 0
    with_validation = 0
    
    for model_name, key in name_to_key.items():
        filepath = os.path.join(csv_dir, f"history_{key}.csv")
        if os.path.exists(filepath):
            available += 1
            try:
                df = pd.read_csv(filepath)
                if df['train_cost'].notna().sum() > 0:
                    with_training += 1
                if df['val_cost'].notna().sum() > 0:
                    with_validation += 1
            except:
                pass
    
    print(f"📁 CSV files available: {available}/{len(name_to_key)}")
    print(f"🔥 Models with training data: {with_training}/{len(name_to_key)}")
    print(f"✨ Models with validation data: {with_validation}/{len(name_to_key)}")
    
    if available == len(name_to_key):
        print("✅ All models have CSV training history files!")
    else:
        print("⚠️ Some models missing CSV files")

if __name__ == "__main__":
    verify_training_data()
