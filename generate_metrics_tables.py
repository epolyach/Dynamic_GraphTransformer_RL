#!/usr/bin/env python3
"""
Generate tabulated CSV files with performance metrics for each solver/model.
Metrics include: cpc, std_cpc, tpi (time_per_instance), std_tpi, is_optimal
"""

import pandas as pd
import numpy as np
from pathlib import Path

def process_benchmark_data(csv_file):
    """
    Process benchmark data and generate metric tables for each solver.
    
    Args:
        csv_file: Path to the benchmark CSV file
    """
    # Read the benchmark data
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get unique solvers
    solvers = df['solver'].unique()
    print(f"Found {len(solvers)} solvers: {', '.join(solvers)}")
    
    # Create output directory if it doesn't exist
    output_dir = Path('results/csv/metrics_tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each solver
    for solver in solvers:
        print(f"\nProcessing {solver}...")
        
        # Filter data for current solver
        solver_df = df[df['solver'] == solver].copy()
        
        # Calculate metrics for each N value
        metrics_list = []
        
        for n in sorted(solver_df['n'].unique()):
            n_data = solver_df[solver_df['n'] == n]
            
            # Calculate metrics
            metrics = {
                'n': n,
                'cpc': n_data['cpc'].mean(),
                'std_cpc': n_data['cpc'].std(),
                'tpi': n_data['time'].mean(),
                'std_tpi': n_data['time'].std(),
                # For optimality, we consider an instance optimal if it wasn't timed out or failed
                'is_optimal': ((~n_data['timeout']) & (~n_data['failed'])).mean()
            }
            
            metrics_list.append(metrics)
        
        # Create DataFrame with metrics
        metrics_df = pd.DataFrame(metrics_list)
        
        # Sort by N value
        metrics_df = metrics_df.sort_values('n').reset_index(drop=True)
        
        # Save to CSV
        output_file = output_dir / f'{solver}_metrics.csv'
        metrics_df.to_csv(output_file, index=False, float_format='%.6f')
        print(f"  Saved metrics to {output_file}")
        
        # Display summary
        print(f"  Summary for {solver}:")
        print(f"    N values: {metrics_df['n'].min()} to {metrics_df['n'].max()}")
        print(f"    Avg CPC: {metrics_df['cpc'].mean():.4f} ± {metrics_df['std_cpc'].mean():.4f}")
        print(f"    Avg TPI: {metrics_df['tpi'].mean():.4f} ± {metrics_df['std_tpi'].mean():.4f}")
        print(f"    Overall optimality rate: {metrics_df['is_optimal'].mean():.2%}")
    
    # Also create a combined table with all solvers side by side
    print("\nCreating combined metrics table...")
    
    combined_data = []
    for solver in solvers:
        solver_df = df[df['solver'] == solver].copy()
        
        for n in sorted(df['n'].unique()):
            n_data = solver_df[solver_df['n'] == n]
            
            if len(n_data) > 0:
                combined_data.append({
                    'n': n,
                    'solver': solver,
                    'cpc': n_data['cpc'].mean(),
                    'std_cpc': n_data['cpc'].std(),
                    'tpi': n_data['time'].mean(),
                    'std_tpi': n_data['time'].std(),
                    'is_optimal': ((~n_data['timeout']) & (~n_data['failed'])).mean()
                })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Pivot to create wide format with solvers as column groups
    pivot_metrics = []
    for metric in ['cpc', 'std_cpc', 'tpi', 'std_tpi', 'is_optimal']:
        pivot = combined_df.pivot(index='n', columns='solver', values=metric)
        pivot.columns = [f'{col}_{metric}' for col in pivot.columns]
        pivot_metrics.append(pivot)
    
    # Combine all pivoted metrics
    combined_wide = pd.concat(pivot_metrics, axis=1)
    
    # Reorder columns to group by solver
    cols = []
    for solver in solvers:
        cols.extend([f'{solver}_cpc', f'{solver}_std_cpc', f'{solver}_tpi', f'{solver}_std_tpi', f'{solver}_is_optimal'])
    
    # Only include columns that exist
    cols = [col for col in cols if col in combined_wide.columns]
    combined_wide = combined_wide[cols]
    
    # Save combined table
    combined_file = output_dir / 'all_solvers_metrics.csv'
    combined_wide.to_csv(combined_file, float_format='%.6f')
    print(f"  Saved combined metrics to {combined_file}")
    
    return output_dir

def display_tables(output_dir):
    """Display the generated tables for verification."""
    print("\n" + "="*80)
    print("GENERATED METRIC TABLES")
    print("="*80)
    
    # List all generated CSV files
    csv_files = sorted(output_dir.glob('*.csv'))
    
    for csv_file in csv_files:
        if 'all_solvers' not in csv_file.name:
            print(f"\n{csv_file.name}")
            print("-" * len(csv_file.name))
            df = pd.read_csv(csv_file)
            print(df.to_string(index=False))
    
    # Show combined table separately
    combined_file = output_dir / 'all_solvers_metrics.csv'
    if combined_file.exists():
        print(f"\n\nCOMBINED TABLE: {combined_file.name}")
        print("=" * 40)
        df = pd.read_csv(combined_file)
        print("First 10 rows:")
        print(df.head(10).to_string())

if __name__ == "__main__":
    # Use the most recent benchmark file
    csv_file = 'results/csv/benchmark_modified_20250827_074153.csv'
    
    # Process data and generate tables
    output_dir = process_benchmark_data(csv_file)
    
    # Display results
    display_tables(output_dir)
    
    print("\n✅ Successfully generated metric tables for all solvers!")
    print(f"📁 Files saved in: {output_dir}")
