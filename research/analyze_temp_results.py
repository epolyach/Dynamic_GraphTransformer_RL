#!/usr/bin/env python3
"""
Comprehensive Temperature Experiment Analysis
"""

import pandas as pd
import numpy as np

def main():
    # Load the temperature experiment results
    df = pd.read_csv('./results/temperature/temperature_experiments_summary.csv')

    print('🔥 COMPREHENSIVE TEMPERATURE EXPERIMENT ANALYSIS')
    print('=' * 60)

    # Overall statistics
    print(f'\n📊 EXPERIMENT OVERVIEW:')
    print(f'   • Total experiments: {len(df)}')
    print(f'   • Models tested: {len(df["model_name"].unique())} ({list(df["model_name"].unique())})')
    print(f'   • Temperature regimes: {len(df["regime_name"].unique())}')
    print(f'   • Seeds per combination: {len(df["seed"].unique())}')

    # Best performance analysis
    print(f'\n🏆 TOP PERFORMANCES BY FINAL VALIDATION COST:')
    top_10 = df.nsmallest(10, 'final_val_cost')[['model_name', 'regime_name', 'seed', 'final_val_cost', 'best_val_cost', 'convergence_epoch', 'training_time']]
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f'   {idx:2d}. {row["model_name"]:15s} | {row["regime_name"]:12s} | seed{row["seed"]:3d} | Final: {row["final_val_cost"]:.4f} | Best: {row["best_val_cost"]:.4f} | Epoch: {row["convergence_epoch"]:2.0f} | Time: {row["training_time"]:5.1f}s')

    # Model analysis
    print(f'\n📈 PERFORMANCE BY MODEL (averaged across all regimes/seeds):')
    model_stats = df.groupby('model_name').agg({
        'final_val_cost': ['mean', 'std', 'min'],
        'best_val_cost': ['mean', 'min'],
        'training_time': ['mean'],
        'total_parameters': 'first',
        'convergence_epoch': ['mean']
    }).round(4)

    for model in model_stats.index:
        print(f'   {model:15s}: Final Cost: {model_stats.loc[model, ("final_val_cost", "mean")]:.4f} ± {model_stats.loc[model, ("final_val_cost", "std")]:.4f} | Best: {model_stats.loc[model, ("best_val_cost", "min")]:.4f} | Params: {int(model_stats.loc[model, ("total_parameters", "first")]):,} | Time: {model_stats.loc[model, ("training_time", "mean")]:.1f}s')

    # Regime analysis
    print(f'\n🌡️ PERFORMANCE BY TEMPERATURE REGIME (averaged across all models/seeds):')
    regime_stats = df.groupby('regime_name').agg({
        'final_val_cost': ['mean', 'std', 'min'],
        'best_val_cost': ['mean', 'min'],
        'convergence_epoch': ['mean']
    }).round(4)

    regime_ranking = regime_stats.sort_values(('final_val_cost', 'mean'))
    for idx, (regime, stats) in enumerate(regime_ranking.iterrows(), 1):
        print(f'   {idx:2d}. {regime:15s}: Final: {stats[("final_val_cost", "mean")]:.4f} ± {stats[("final_val_cost", "std")]:.4f} | Best: {stats[("best_val_cost", "min")]:.4f} | Epochs: {stats[("convergence_epoch", "mean")]:.1f}')

    # Temperature regime + model combinations
    print(f'\n🔥 BEST MODEL-REGIME COMBINATIONS:')
    combo_stats = df.groupby(['model_name', 'regime_name']).agg({
        'final_val_cost': ['mean', 'std', 'min'],
        'best_val_cost': 'min',
        'training_time': 'mean'
    }).round(4)

    combo_ranking = combo_stats.sort_values(('final_val_cost', 'mean')).head(15)
    for idx, ((model, regime), stats) in enumerate(combo_ranking.iterrows(), 1):
        print(f'   {idx:2d}. {model:15s} + {regime:12s}: {stats[("final_val_cost", "mean")]:.4f} ± {stats[("final_val_cost", "std")]:.4f} | Best: {stats[("best_val_cost", "min")]:.4f}')

    # Statistical analysis
    print(f'\n🧮 STATISTICAL INSIGHTS:')
    overall_mean = df['final_val_cost'].mean()
    overall_std = df['final_val_cost'].std()
    cv = overall_std / overall_mean
    print(f'   • Overall mean final cost: {overall_mean:.4f} ± {overall_std:.4f}')
    print(f'   • Coefficient of variation: {cv:.3f} ({cv*100:.1f}%)')
    print(f'   • Performance range: {df["final_val_cost"].min():.4f} - {df["final_val_cost"].max():.4f}')
    print(f'   • Improvement from worst to best: {((df["final_val_cost"].max() - df["final_val_cost"].min()) / df["final_val_cost"].max() * 100):.1f}%')

    # Parameter efficiency analysis
    print(f'\n⚡ PARAMETER EFFICIENCY ANALYSIS:')
    df['efficiency'] = 1 / (df['final_val_cost'] * df['total_parameters'])
    eff_stats = df.groupby('model_name')['efficiency'].agg(['mean', 'std']).round(2)
    eff_ranking = eff_stats.sort_values('mean', ascending=False)
    for model, stats in eff_ranking.iterrows():
        print(f'   {model:15s}: {stats["mean"]:.2e} ± {stats["std"]:.2e} (1/(cost × params))')

    # Temperature regime effectiveness
    print(f'\n🌡️ TEMPERATURE REGIME EFFECTIVENESS:')
    print('   (Regimes ranked by average final validation cost across all models)')
    regime_effectiveness = df.groupby('regime_name')['final_val_cost'].agg(['mean', 'count']).sort_values('mean')
    for idx, (regime, stats) in enumerate(regime_effectiveness.iterrows(), 1):
        count = int(stats['count'])
        mean_cost = stats['mean']
        print(f'   {idx:2d}. {regime:15s}: {mean_cost:.4f} (n={count})')
    
    # Key insights
    print(f'\n💡 KEY INSIGHTS:')
    best_overall = df.loc[df['final_val_cost'].idxmin()]
    print(f'   • Best overall performance: {best_overall["model_name"]} with {best_overall["regime_name"]} regime')
    print(f'     Final cost: {best_overall["final_val_cost"]:.4f}, Training time: {best_overall["training_time"]:.1f}s')
    
    # Model comparison
    best_by_model = df.groupby('model_name')['final_val_cost'].min().sort_values()
    print(f'   • Best performance by model:')
    for model, best_cost in best_by_model.items():
        params = df[df['model_name'] == model]['total_parameters'].iloc[0]
        print(f'     {model:15s}: {best_cost:.4f} ({params:,} parameters)')

if __name__ == "__main__":
    main()
