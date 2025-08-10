"""
Comprehensive training plots using only real data, no mocks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, Any
import json
import re


def load_comprehensive_data_for_plots(df: pd.DataFrame, results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load comprehensive data with strict validation - NO FALLBACKS."""
    results = {}
    
    for _, row in df.iterrows():
        model_name = row['Model']
        val_cost_per_customer = row['Val/Cust']
        training_time = row.get('CPU Time (s)', 0.0) or 0.0
        
        model_results = {
            'train_losses': [],
            'train_costs': [],
            'val_costs': [],
            'training_times': [],
            'final_val_cost': val_cost_per_customer * 20,
            'total_training_time': training_time,
            'model_parameters': None,
            'out_dir': None,
        }
        
        outdir = row.get('OutDir')
        if isinstance(outdir, str) and len(outdir) > 0:
            model_results['out_dir'] = outdir
            history_path = Path(outdir) / "train_history.csv"
            
            if not history_path.exists():
                raise FileNotFoundError(f"Missing train_history.csv for {model_name}: {history_path}")
            
            try:
                df_history = pd.read_csv(history_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read {history_path}: {e}")
            
            # Strict column validation
            required_cols = ['epoch', 'train_time_s', 'train_loss', 'train_cost_per_customer', 'val_cost_per_customer']
            missing_cols = [col for col in required_cols if col not in df_history.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {history_path}: {missing_cols}. Found: {list(df_history.columns)}")
            
            if df_history.empty:
                raise ValueError(f"Empty train_history.csv for {model_name}: {history_path}")
            
            # Load data with strict validation
            epochs = sorted(df_history['epoch'].dropna().unique())
            if not epochs:
                raise ValueError(f"No valid epochs in {history_path}")
            
            for epoch in epochs:
                epoch_data = df_history[df_history['epoch'] == epoch]
                
                # Training time per epoch
                epoch_time = epoch_data['train_time_s'].mean()
                if pd.isna(epoch_time):
                    raise ValueError(f"Missing train_time_s for epoch {epoch} in {model_name}")
                model_results['training_times'].append(float(epoch_time))
                
                # Training loss per epoch
                loss_epoch = epoch_data['train_loss'].mean()
                if pd.isna(loss_epoch):
                    raise ValueError(f"Missing train_loss for epoch {epoch} in {model_name}")
                model_results['train_losses'].append(float(loss_epoch))
                
                # Training cost per epoch
                train_cost_epoch = epoch_data['train_cost_per_customer'].mean()
                if pd.isna(train_cost_epoch):
                    raise ValueError(f"Missing train_cost_per_customer for epoch {epoch} in {model_name}")
                model_results['train_costs'].append(float(train_cost_epoch) * 20)
                
                # Validation cost per epoch
                val_cost_epoch = epoch_data['val_cost_per_customer'].mean()
                if pd.isna(val_cost_epoch):
                    raise ValueError(f"Missing val_cost_per_customer for epoch {epoch} in {model_name}")
                model_results['val_costs'].append(float(val_cost_epoch) * 20)
            
            # Try to calculate model parameters
            try:
                from src.training.pipeline_train import get_model
                device = torch.device('cpu')
                model = get_model(model_name, device)
                model_results['model_parameters'] = sum(p.numel() for p in model.parameters())
            except Exception as e:
                print(f"Could not load model {model_name} for parameter counting: {e}")
                model_results['model_parameters'] = None
        else:
            # For baseline models, no training history expected
            if model_name not in ['naive_baseline', 'greedy_baseline']:
                raise ValueError(f"Missing OutDir for trainable model {model_name}")
            model_results['model_parameters'] = 0
        
        results[model_name] = model_results
    
    return results

def create_comprehensive_training_plots(results: Dict[str, Dict], output_path: Path, 
                                       customers: int = 20):
    """Create comprehensive comparison plots using only real data."""
    
    if not results:
        print("No training results found for plotting.")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    model_names = list(results.keys())
    
    # 1. Training Cost Evolution (Per Customer)
    plt.subplot(2, 4, 1)
    any_curve = False
    for i, (model_name, result) in enumerate(results.items()):
        if result['train_costs']:
            any_curve = True
            costs_pc = [cost / customers for cost in result['train_costs']]
            epochs = list(range(len(costs_pc)))
            plt.plot(epochs, costs_pc, label=model_name, linewidth=2, marker='o', markersize=3, color=colors[i % len(colors)])
    if any_curve:
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Training history not found or empty', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.title('Training Loss Evolution', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 2. Training Cost Evolution (Per Customer)
    plt.subplot(2, 4, 2)
    any_train = False
    for i, (model_name, result) in enumerate(results.items()):
        if result['train_costs']:
            any_train = True
            vals_pc = [cost / customers for cost in result['train_costs']]
            epochs = list(range(len(vals_pc)))
            plt.plot(epochs, vals_pc, label=model_name, linewidth=2, marker='o', markersize=3, color=colors[i % len(colors)])
    if any_train:
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Training history not found or empty', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.title('Training Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Cost per Customer')
    plt.grid(True, alpha=0.3)

    # 3. Validation Cost Evolution (Per Customer) - top row third panel
    plt.subplot(2, 4, 3)
    any_val = False
    for i, (model_name, result) in enumerate(results.items()):
        if result['val_costs']:
            any_val = True
            vals_pc = [cost / customers for cost in result['val_costs']]
            epochs = list(range(len(vals_pc)))
            plt.plot(epochs, vals_pc, label=model_name, linewidth=2, marker='s', markersize=3, color=colors[i % len(colors)])
    if any_val:
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Validation history not found or empty', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.title('Validation Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Avg Cost per Customer')
    plt.grid(True, alpha=0.3)
        # 5. Training Time (per epoch)
    plt.subplot(2, 4, 5)
    for i, (model_name, result) in enumerate(results.items()):
        if result['training_times']:
            epochs = list(range(len(result['training_times'])))
            plt.plot(epochs, result['training_times'], 'o-', label=model_name, linewidth=2, markersize=4, color=colors[i % len(colors)])
    plt.title('Training Time (per epoch)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart - REAL DATA
    plt.subplot(2, 4, 4)
    final_costs_normalized = [results[name]['final_val_cost'] / customers for name in model_names]
    
    bars = plt.bar(range(len(model_names)), final_costs_normalized, color=colors[:len(model_names)], alpha=0.8)
    plt.title('Final Performance (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Average Cost per Customer')
    plt.xticks(range(len(model_names)), [name.replace('_', '\n') for name in model_names], rotation=0)
    
    for bar, cost in zip(bars, final_costs_normalized):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{cost:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance vs Complexity (Final Cost per Customer vs Parameters)
    plt.subplot(2, 4, 8)
    pts = []
    for name in model_names:
        params = results[name]['model_parameters']
        final_cost_pc = (results[name]['final_val_cost'] / customers) if results[name].get('final_val_cost') is not None else None
        if params is not None and final_cost_pc is not None:
            pts.append((name, params, final_cost_pc))
    if pts:
        for i, (name, params, cost_pc) in enumerate(pts):
            plt.scatter(params, cost_pc, s=120, color=colors[i % len(colors)], alpha=0.85, label=name)
            plt.annotate(name.replace('_', '\n'), (params, cost_pc), xytext=(5,5), textcoords='offset points', fontsize=9)
        plt.title('Performance vs Complexity', fontsize=12, fontweight='bold')
        plt.xlabel('Parameters')
        plt.ylabel('Final Cost per Customer')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No parameter/performance data', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # 6. Model Complexity - REAL DATA where available
    plt.subplot(2, 4, 6)
    param_counts = []
    available_models = []
    for name in model_names:
        if results[name]['model_parameters'] is not None:
            param_counts.append(results[name]['model_parameters'])
            available_models.append(name)
    
    if param_counts:
        bars = plt.bar(range(len(available_models)), param_counts, color=colors[:len(available_models)], alpha=0.8)
        plt.title('Model Complexity (Real Parameters)', fontsize=12, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Parameters')
        plt.xticks(range(len(available_models)), [name.replace('_', '\n') for name in available_models])
        
        for bar, params in zip(bars, param_counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(param_counts)*0.01,
                    f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'Model Parameter Data\nNot Available\n\n(Could not load models\nfor parameter counting)', 
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.subplot(2, 4, 7)
    import numpy as _np
    print('[LE] Computing Learning Efficiency (improvement from first to last train_cost)')
    improvements = []
    names = []
    for model_name in model_names:
        tc = results[model_name].get('train_costs', [])
        print(f"[LE] {model_name}: train_costs_len={len(tc)}")
        if tc and len(tc) >= 2:
            initial_cost = float(tc[0])
            final_cost = float(tc[-1])
            print(f"[LE] {model_name}: initial={initial_cost}, final={final_cost}")
            if not (_np.isfinite(initial_cost) and _np.isfinite(final_cost)):
                print(f"[LE][WARN] Non-finite values for {model_name}: initial={initial_cost}, final={final_cost}")
                continue
            improvement = 0.0 if initial_cost == 0.0 else ((initial_cost - final_cost) / initial_cost) * 100.0
            if not _np.isfinite(improvement):
                print(f"[LE][WARN] Non-finite improvement for {model_name}: {improvement}")
                continue
            improvements.append(float(improvement))
            names.append(model_name)
            print(f"[LE] {model_name}: improvement={improvement:.3f}%")
        else:
            print(f"[LE] {model_name}: insufficient train_costs for improvement computation")
    if improvements:
        bars = plt.bar(range(len(names)), improvements, color=colors[:len(names)], alpha=0.8)
        plt.title('Learning Efficiency', fontsize=12, fontweight='bold')
        plt.xlabel('Model Architecture')
        plt.ylabel('Cost Improvement (%)')
        plt.xticks(range(len(names)), [n.replace('_', '\n') for n in names])
        ypad = max(0.2, 0.01*max(improvements))
        for bar, imp in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + ypad, f'{imp:.1f}%',
                     ha='center', va='bottom', fontweight='bold')
    else:
        print('[LE] No valid improvements computed; panel will show placeholder')
        plt.text(0.5, 0.5, 'Learning Efficiency\nNot Available', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.grid(True, alpha=0.3, axis='y')

    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Real data comparison plots saved to {output_path}")
    plt.close()
