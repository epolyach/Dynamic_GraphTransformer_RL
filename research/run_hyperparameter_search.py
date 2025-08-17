
import yaml
import subprocess
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import time

def create_breakthrough_analysis(results_df, target_cost=0.5):
    """Analyze results specifically for breakthrough performance."""
    breakthrough_results = results_df[results_df['validation_cost'] < target_cost]
    print(f"\n🎯 BREAKTHROUGH ANALYSIS (target < {target_cost}):")
    print(f"   Breakthrough experiments: {len(breakthrough_results)}/{len(results_df)} ({len(breakthrough_results)/len(results_df)*100:.1f}%)")
    
    if len(breakthrough_results) > 0:
        print("\n🏆 BREAKTHROUGH CONFIGURATIONS:")
        print("-" * 100)
        for i, (_, row) in enumerate(breakthrough_results.nsmallest(10, 'validation_cost').iterrows(), 1):
            print(f"{i:2d}. {row['model']:12} | Cost: {row['validation_cost']:.4f} | "
                  f"Emb: {row['embedding_dim']:4} | Layers: {row['n_layers']:2} | Heads: {row['n_heads']:2} | "
                  f"LR: {row['learning_rate']:.1e} | Drop: {row['dropout']:.1f} | "
                  f"Batch: {row.get('batch_size', 'N/A'):4} | Temp: {row.get('temp_start', 'N/A')}/{row.get('temp_min', 'N/A')}")
        
        # Analyze breakthrough patterns
        print("\n📊 BREAKTHROUGH PATTERNS:")
        print("-" * 50)
        for param in ['model', 'embedding_dim', 'n_layers', 'n_heads', 'learning_rate', 'dropout']:
            if param in breakthrough_results.columns:
                top_values = breakthrough_results[param].value_counts().head(3)
                print(f"{param:15}: {dict(top_values)}")
    else:
        print("   ❌ No breakthrough results yet. Keep searching!")
        
        # Show closest attempts
        closest = results_df.nsmallest(5, 'validation_cost')
        print("\n🔍 CLOSEST ATTEMPTS:")
        print("-" * 80)
        for i, (_, row) in enumerate(closest.iterrows(), 1):
            gap = row['validation_cost'] - target_cost
            print(f"{i}. {row['model']:12} | Cost: {row['validation_cost']:.4f} | Gap: +{gap:.4f} | "
                  f"Config: {row['embedding_dim']}/{row['n_layers']}/{row['n_heads']}")
    
    return len(breakthrough_results)

def create_incremental_plots(results_df, experiment_count, total_experiments):
    """Create plots showing current progress and results."""
    if len(results_df) == 0:
        return
    
    # Create plots directory
    plots_dir = 'results/hyperparameter_search_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive progress plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title with progress information and breakthrough status
    completion_pct = (experiment_count / total_experiments) * 100
    breakthrough_count = len(results_df[results_df['validation_cost'] < 0.5])
    fig.suptitle(f'Hyperparameter Search Progress: {experiment_count}/{total_experiments} ({completion_pct:.1f}%)\n'
                f'🎯 Target <0.5 achieved: {breakthrough_count} experiments | Generated at: {datetime.now().strftime("%H:%M:%S")}',
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Current best performance by model (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    model_best = results_df.groupby('model')['validation_cost'].min().sort_values()
    bars = ax1.bar(range(len(model_best)), model_best.values, alpha=0.7)
    ax1.set_title('Best Performance by Model', fontweight='bold')
    ax1.set_ylabel('Validation Cost')
    ax1.set_xticks(range(len(model_best)))
    ax1.set_xticklabels(model_best.index, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, model_best.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Performance distribution (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax2.hist(model_data['validation_cost'], alpha=0.6, label=model, bins=15)
    ax2.set_title('Performance Distribution', fontweight='bold')
    ax2.set_xlabel('Validation Cost')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate vs performance (middle-left)
    ax3 = fig.add_subplot(gs[0, 2])
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax3.scatter(model_data['learning_rate'], model_data['validation_cost'], 
                   alpha=0.6, label=model, s=50)
    ax3.set_title('Learning Rate vs Performance', fontweight='bold')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Validation Cost')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Embedding dimension vs performance (middle-right)
    ax4 = fig.add_subplot(gs[0, 3])
    embedding_stats = results_df.groupby('embedding_dim')['validation_cost'].agg(['mean', 'std']).reset_index()
    bars = ax4.bar(embedding_stats['embedding_dim'], embedding_stats['mean'], 
                  yerr=embedding_stats['std'], capsize=5, alpha=0.7)
    ax4.set_title('Embedding Dimension vs Performance', fontweight='bold')
    ax4.set_xlabel('Embedding Dimension')
    ax4.set_ylabel('Validation Cost (mean ± std)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Number of layers analysis (middle-left)
    ax5 = fig.add_subplot(gs[1, 0])
    layer_stats = results_df.groupby('n_layers')['validation_cost'].agg(['mean', 'std']).reset_index()
    bars = ax5.bar(layer_stats['n_layers'], layer_stats['mean'], 
                  yerr=layer_stats['std'], capsize=5, alpha=0.7, color='green')
    ax5.set_title('Number of Layers vs Performance', fontweight='bold')
    ax5.set_xlabel('Number of Layers')
    ax5.set_ylabel('Validation Cost (mean ± std)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Attention heads analysis (middle-right)
    ax6 = fig.add_subplot(gs[1, 1])
    heads_stats = results_df.groupby('n_heads')['validation_cost'].agg(['mean', 'std']).reset_index()
    bars = ax6.bar(heads_stats['n_heads'], heads_stats['mean'], 
                  yerr=heads_stats['std'], capsize=5, alpha=0.7, color='orange')
    ax6.set_title('Attention Heads vs Performance', fontweight='bold')
    ax6.set_xlabel('Number of Attention Heads')
    ax6.set_ylabel('Validation Cost (mean ± std)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Dropout analysis (middle-right-2)
    ax7 = fig.add_subplot(gs[1, 2])
    dropout_stats = results_df.groupby('dropout')['validation_cost'].agg(['mean', 'std']).reset_index()
    bars = ax7.bar(dropout_stats['dropout'], dropout_stats['mean'], 
                  yerr=dropout_stats['std'], capsize=5, alpha=0.7, color='red')
    ax7.set_title('Dropout vs Performance', fontweight='bold')
    ax7.set_xlabel('Dropout Rate')
    ax7.set_ylabel('Validation Cost (mean ± std)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Top 10 configurations (middle-right-3)
    ax8 = fig.add_subplot(gs[1, 3])
    top_10 = results_df.nsmallest(10, 'validation_cost')
    y_pos = range(len(top_10))
    bars = ax8.barh(y_pos, top_10['validation_cost'], alpha=0.7, color='purple')
    ax8.set_title('Top 10 Configurations', fontweight='bold')
    ax8.set_xlabel('Validation Cost')
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([f"{row['model']}\n{row['embedding_dim']}/{row['n_layers']}/{row['n_heads']}" 
                        for _, row in top_10.iterrows()], fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistical summary (bottom row)
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis('off')
    
    # Create statistical summary
    summary_text = f"""HYPERPARAMETER SEARCH PROGRESS SUMMARY

Experiments Completed: {experiment_count}/{total_experiments} ({completion_pct:.1f}%)
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Best Results:
"""
    
    for model in sorted(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        if len(model_data) > 0:
            best_result = model_data.loc[model_data['validation_cost'].idxmin()]
            summary_text += f"\n{model}:\n"
            summary_text += f"  • Best Cost: {best_result['validation_cost']:.4f}\n"
            summary_text += f"  • Config: emb={best_result['embedding_dim']}, layers={best_result['n_layers']}, "
            summary_text += f"heads={best_result['n_heads']}, lr={best_result['learning_rate']:.1e}, dropout={best_result['dropout']}\n"
            summary_text += f"  • Experiments: {len(model_data)}\n"
    
    overall_best = results_df.loc[results_df['validation_cost'].idxmin()]
    summary_text += f"\nOverall Best: {overall_best['model']} with cost {overall_best['validation_cost']:.4f}"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"hyperparameter_search_progress_{experiment_count}_{timestamp}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also save a "latest" version for easy access
    latest_path = os.path.join(plots_dir, "hyperparameter_search_latest.png")
    plt.figure(figsize=(20, 12))
    plt.subplot(111)
    plt.imshow(plt.imread(plot_path))
    plt.axis('off')
    plt.savefig(latest_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📈 Progress plot saved: {plot_path}")
    print(f"📈 Latest plot updated: {latest_path}")

def run_hyperparameter_search():
    with open('configs/hyperparameter_search.yaml', 'r') as f:
        config = yaml.safe_load(f)

    hyperparameters = config['hyperparameters']
    models = config['models']

    search_space = list(itertools.product(*hyperparameters.values()))
    results = []
    total_experiments = len(search_space) * len(models)
    experiment_count = 0
    
    print(f"🚀 Starting hyperparameter search with {total_experiments} total experiments")
    print(f"📊 Search space: {len(search_space)} parameter combinations × {len(models)} models")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    for i, params in enumerate(search_space):
        param_dict = dict(zip(hyperparameters.keys(), params))
        
        print(f"\n🔬 Parameter set {i+1}/{len(search_space)}: {param_dict}")

        for j, model in enumerate(models):
            command = [
                'python',
                'run_experimental_training.py',
                '--models', model,
                '--config', 'configs/hyperparameter_search.yaml',
                '--embedding_dim', str(param_dict['embedding_dim']),
                '--n_layers', str(param_dict['n_layers']),
                '--n_heads', str(param_dict['n_heads']),
                '--learning_rate', str(param_dict['learning_rate']),
                '--dropout', str(param_dict['dropout'])
            ]
            
            # Add additional parameters if they exist
            if 'batch_size' in param_dict:
                command.extend(['--batch_size', str(param_dict['batch_size'])])
            if 'temp_start' in param_dict:
                command.extend(['--temp_start', str(param_dict['temp_start'])])
            if 'temp_min' in param_dict:
                command.extend(['--temp_min', str(param_dict['temp_min'])])
            if 'temp_decay' in param_dict:
                command.extend(['--temp_decay', str(param_dict['temp_decay'])])

            experiment_count += 1
            start_time = time.time()
            print(f"  🤖 Running {model} (experiment {experiment_count}/{total_experiments})... ", end="")
            
            try:
                process_result = subprocess.run(command, check=True, capture_output=True, text=True)
                output = process_result.stdout
                elapsed_time = time.time() - start_time

                # Create the directory and save the log
                model_log_dir = f'results/default/{model}'
                os.makedirs(model_log_dir, exist_ok=True)
                with open(f'{model_log_dir}/training_log.txt', 'w') as log_file:
                    log_file.write(output)

                # Parse the validation cost from the output
                val_cost = float('nan')
                for line in reversed(output.strip().split('\n')):
                    if 'final validation cost' in line:
                        val_cost = float(line.split(':')[1].strip())
                        break

                result = param_dict.copy()
                result['model'] = model
                result['validation_cost'] = val_cost
                result['experiment_id'] = experiment_count
                result['timestamp'] = datetime.now().isoformat()
                results.append(result)
                
                # Breakthrough detection
                breakthrough_status = "🎯 BREAKTHROUGH!" if val_cost < 0.5 else "✅"
                print(f"{breakthrough_status} Cost: {val_cost:.4f} ({elapsed_time:.1f}s)")
                
                # Save incremental results
                df_current = pd.DataFrame(results)
                df_current.to_csv('results/hyperparameter_search_results_incremental.csv', index=False)
                
                # Generate plots every 5 experiments, after breakthrough, or for the last few experiments
                breakthrough_achieved = val_cost < 0.5
                if experiment_count % 5 == 0 or breakthrough_achieved or experiment_count >= total_experiments - 10:
                    create_incremental_plots(df_current, experiment_count, total_experiments)
                
                # Run breakthrough analysis every 10 experiments or immediately after breakthrough
                if experiment_count % 10 == 0 or breakthrough_achieved:
                    breakthrough_count = create_breakthrough_analysis(df_current)
                    if breakthrough_count > 0:
                        print(f"\n🔥 Found {breakthrough_count} breakthrough configurations so far!")

            except subprocess.CalledProcessError as e:
                elapsed_time = time.time() - start_time
                print(f"❌ Failed ({elapsed_time:.1f}s)")
                print(f"   Error running {model} with params {param_dict}: {e}")
                if e.stderr:
                    print(f"   Stderr: {e.stderr[:200]}...")  # Limit error output

    # Final results processing
    df = pd.DataFrame(results)
    df.to_csv('results/hyperparameter_search_results.csv', index=False)
    
    # Generate final comprehensive plots and analysis
    if len(results) > 0:
        create_incremental_plots(df, len(results), total_experiments)
        
        # Final breakthrough analysis
        final_breakthrough_count = create_breakthrough_analysis(df)
        
        print(f"\n🎉 Hyperparameter search complete!")
        print(f"📊 Total experiments completed: {len(results)}/{total_experiments}")
        print(f"🎯 Breakthrough experiments (<0.5): {final_breakthrough_count}")
        print(f"📁 Results saved to: results/hyperparameter_search_results.csv")
        print(f"📈 Final plots saved to: results/hyperparameter_search_plots/")
        
        # Print top results summary
        print("\n🏆 TOP 10 RESULTS:")
        print("-" * 100)
        top_10 = df.nsmallest(10, 'validation_cost')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            breakthrough_marker = "🎯" if row['validation_cost'] < 0.5 else "  "
            print(f"{breakthrough_marker}{i:2d}. {row['model']:12} | Cost: {row['validation_cost']:.4f} | "
                  f"Emb: {row['embedding_dim']:4} | Layers: {row['n_layers']:2} | Heads: {row['n_heads']:2} | "
                  f"LR: {row['learning_rate']:.1e} | Drop: {row['dropout']:.1f}")
    else:
        print("⚠️ No successful experiments completed.")

if __name__ == '__main__':
    run_hyperparameter_search()

