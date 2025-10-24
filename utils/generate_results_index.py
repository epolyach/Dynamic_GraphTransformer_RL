#!/usr/bin/env python3
"""Generate comprehensive index of all consolidated results."""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def get_file_info(filepath):
    """Get file size and modification time."""
    stat = filepath.stat()
    return {
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    }

def analyze_training_result(result_dir):
    """Analyze a training result directory."""
    info = {
        'name': result_dir.name,
        'path': str(result_dir.relative_to('results_consolidated')),
        'has_csv': False,
        'has_json': False,
        'has_model': False,
        'best_cpc': None,
        'final_cpc': None,
        'epochs': None
    }
    
    # Check for CSV
    csv_file = result_dir / 'csv' / 'history_gt_rl.csv'
    if csv_file.exists():
        info['has_csv'] = True
        try:
            df = pd.read_csv(csv_file)
            if 'Val CPC' in df.columns:
                info['best_cpc'] = float(df['Val CPC'].min())
                info['final_cpc'] = float(df['Val CPC'].iloc[-1])
                info['epochs'] = len(df)
        except Exception as e:
            print(f"  Warning: Could not read {csv_file}: {e}")
    
    # Check for JSON
    json_file = result_dir / 'GT_RL_training_summary.json'
    if json_file.exists():
        info['has_json'] = True
        try:
            with open(json_file) as f:
                data = json.load(f)
                if 'best_validation_cpc' in data:
                    info['best_cpc'] = data['best_validation_cpc']
        except Exception as e:
            print(f"  Warning: Could not read {json_file}: {e}")
    
    # Check for model
    model_files = list(result_dir.glob('**/*.pth'))
    if model_files:
        info['has_model'] = True
    
    return info

def main():
    """Generate results index."""
    print("Generating results index...")
    
    base_path = Path('results_consolidated')
    
    markdown = ["# Consolidated Results Index\n"]
    markdown.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    markdown.append(f"**Total Size:** {sum(f.stat().st_size for f in base_path.rglob('*') if f.is_file()) / (1024**3):.2f} GB\n")
    markdown.append("\n---\n\n")
    
    # Training Results - GPU
    markdown.append("## Training Results - GPU\n\n")
    markdown.append("### From training_gpu_prefetch (Most Recent)\n\n")
    
    prefetch_dir = base_path / 'training' / 'gpu' / 'from_prefetch'
    if prefetch_dir.exists():
        experiments = sorted([d for d in prefetch_dir.iterdir() if d.is_dir()])
        markdown.append("| Experiment | Epochs | Best CPC | Final CPC | CSV | JSON | Model |\n")
        markdown.append("|------------|--------|----------|-----------|-----|------|-------|\n")
        
        for exp_dir in experiments:
            info = analyze_training_result(exp_dir)
            markdown.append(f"| {info['name']} | {info['epochs'] or 'N/A'} | "
                          f"{info['best_cpc']:.4f if info['best_cpc'] else 'N/A'} | "
                          f"{info['final_cpc']:.4f if info['final_cpc'] else 'N/A'} | "
                          f"{'✅' if info['has_csv'] else '❌'} | "
                          f"{'✅' if info['has_json'] else '❌'} | "
                          f"{'✅' if info['has_model'] else '❌'} |\n")
    
    markdown.append("\n### From training_gpu (Old)\n\n")
    old_gpu_dir = base_path / 'training' / 'gpu' / 'from_training_gpu'
    if old_gpu_dir.exists():
        experiments = sorted([d for d in old_gpu_dir.iterdir() if d.is_dir()])
        markdown.append("| Experiment | Epochs | Best CPC | Final CPC | CSV | JSON | Model |\n")
        markdown.append("|------------|--------|----------|-----------|-----|------|-------|\n")
        
        for exp_dir in experiments:
            info = analyze_training_result(exp_dir)
            markdown.append(f"| {info['name']} | {info['epochs'] or 'N/A'} | "
                          f"{info['best_cpc']:.4f if info['best_cpc'] else 'N/A'} | "
                          f"{info['final_cpc']:.4f if info['final_cpc'] else 'N/A'} | "
                          f"{'✅' if info['has_csv'] else '❌'} | "
                          f"{'✅' if info['has_json'] else '❌'} | "
                          f"{'✅' if info['has_model'] else '❌'} |\n")
    
    # Training Results - CPU
    markdown.append("\n## Training Results - CPU\n\n")
    cpu_dir = base_path / 'training' / 'cpu'
    if cpu_dir.exists():
        experiments = sorted([d for d in cpu_dir.iterdir() if d.is_dir()])
        if experiments:
            markdown.append("| Experiment | Epochs | Best CPC | Final CPC | CSV |\n")
            markdown.append("|------------|--------|----------|-----------|-----|\n")
            
            for exp_dir in experiments:
                info = analyze_training_result(exp_dir)
                markdown.append(f"| {info['name']} | {info['epochs'] or 'N/A'} | "
                              f"{info['best_cpc']:.4f if info['best_cpc'] else 'N/A'} | "
                              f"{info['final_cpc']:.4f if info['final_cpc'] else 'N/A'} | "
                              f"{'✅' if info['has_csv'] else '❌'} |\n")
    
    # Benchmark Results
    markdown.append("\n## Benchmark Results\n\n")
    
    markdown.append("### CPU Benchmarks\n\n")
    cpu_bench_dir = base_path / 'benchmarks' / 'cpu'
    if cpu_bench_dir.exists():
        json_files = list(cpu_bench_dir.rglob('*.json'))
        csv_files = list(cpu_bench_dir.rglob('*.csv'))
        markdown.append(f"- **JSON files:** {len(json_files)}\n")
        markdown.append(f"- **CSV files:** {len(csv_files)}\n")
        markdown.append(f"- **Total size:** {sum(f.stat().st_size for f in cpu_bench_dir.rglob('*') if f.is_file()) / (1024**2):.1f} MB\n\n")
        
        # List major result sets
        result_dirs = sorted([d for d in cpu_bench_dir.iterdir() if d.is_dir()])
        if result_dirs:
            markdown.append("**Result Sets:**\n")
            for rd in result_dirs:
                size = sum(f.stat().st_size for f in rd.rglob('*') if f.is_file()) / (1024**2)
                markdown.append(f"- `{rd.name}` ({size:.1f} MB)\n")
    
    markdown.append("\n### GPU Benchmarks\n\n")
    gpu_bench_dir = base_path / 'benchmarks' / 'gpu'
    if gpu_bench_dir.exists():
        json_files = list(gpu_bench_dir.rglob('*.json'))
        csv_files = list(gpu_bench_dir.rglob('*.csv'))
        markdown.append(f"- **JSON files:** {len(json_files)}\n")
        markdown.append(f"- **CSV files:** {len(csv_files)}\n")
        markdown.append(f"- **Total size:** {sum(f.stat().st_size for f in gpu_bench_dir.rglob('*') if f.is_file()) / (1024**2):.1f} MB\n\n")
        
        # List major result sets
        result_dirs = sorted([d for d in gpu_bench_dir.iterdir() if d.is_dir()])
        if result_dirs:
            markdown.append("**Result Sets:**\n")
            for rd in result_dirs:
                size = sum(f.stat().st_size for f in rd.rglob('*') if f.is_file()) / (1024**2)
                markdown.append(f"- `{rd.name}` ({size:.1f} MB)\n")
    
    # Legacy Results
    markdown.append("\n## Legacy Results (Root Level)\n\n")
    legacy_dir = base_path / 'training' / 'legacy_root_results'
    if legacy_dir.exists():
        experiments = sorted([d for d in legacy_dir.iterdir() if d.is_dir()])
        if experiments:
            markdown.append("| Experiment | Size (MB) | Files |\n")
            markdown.append("|------------|-----------|-------|\n")
            
            for exp_dir in experiments:
                size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file()) / (1024**2)
                files = len(list(exp_dir.rglob('*')))
                markdown.append(f"| {exp_dir.name} | {size:.1f} | {files} |\n")
    
    # Write index
    index_file = base_path / 'RESULTS_INDEX.md'
    with open(index_file, 'w') as f:
        f.write(''.join(markdown))
    
    print(f"✅ Results index created: {index_file}")
    print(f"   Total results documented: {len(experiments) if 'experiments' in locals() else 'N/A'}")

if __name__ == '__main__':
    main()
