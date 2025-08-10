#!/usr/bin/env python3
"""
Standalone Route Visualization Script

This script loads existing test results from the comparative study and generates
route visualizations without re-running the training. Useful for:
- Re-generating plots with different settings
- Creating visualizations after training is complete
- Analyzing existing results without full re-run

Usage:
    python run_visualization_standalone.py [options]
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
import json
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate route visualizations from existing results')
    parser.add_argument('--results_path', type=str, 
                       default='results/small/analysis/test_instance_analysis.pt',
                       help='Path to test instance analysis results file')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'production'],
                       default='small', help='Problem scale (determines output paths)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Custom output directory (overrides scale-based path)')
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Regenerate visualizations even if they already exist')
    parser.add_argument('--html_only', action='store_true',
                       help='Generate only HTML report, skip individual plots')
    parser.add_argument('--plots_only', action='store_true',
                       help='Generate only individual plots, skip HTML report')
    return parser.parse_args()

def load_test_results(results_path, logger):
    """Load test instance analysis results from file"""
    logger.info(f"📂 Loading test results from: {results_path}")
    
    if not os.path.exists(results_path):
        logger.error(f"❌ Results file not found: {results_path}")
        logger.info("💡 Available result files:")
        
        # Look for alternative result files
        base_dir = os.path.dirname(results_path)
        if os.path.exists(base_dir):
            for file in os.listdir(base_dir):
                if file.endswith('.pt') and 'test' in file.lower():
                    logger.info(f"   • {os.path.join(base_dir, file)}")
        
        # Also check for JSON version
        json_path = results_path.replace('.pt', '.json')
        if os.path.exists(json_path):
            logger.info(f"💡 Found JSON version: {json_path}")
            logger.info("   Loading from JSON...")
            
            with open(json_path, 'r') as f:
                test_analysis = json.load(f)
            
            # Convert lists back to numpy arrays for instance data
            test_analysis['test_instance']['coords'] = np.array(test_analysis['test_instance']['coords'])
            test_analysis['test_instance']['demands'] = np.array(test_analysis['test_instance']['demands'])
            
            return test_analysis
        
        sys.exit(1)
    
    try:
        # Try with weights_only=False for compatibility with analysis files
        test_analysis = torch.load(results_path, map_location='cpu', weights_only=False)
        logger.info(f"✅ Successfully loaded test results")
        logger.info(f"   Test instance: {test_analysis['test_instance']['num_customers']} customers")
        logger.info(f"   Models tested: {list(test_analysis['model_results'].keys())}")
        return test_analysis
    
    except Exception as e:
        logger.error(f"❌ Failed to load results: {e}")
        sys.exit(1)

def validate_test_analysis(test_analysis, logger):
    """Validate that test analysis contains required data"""
    required_keys = ['test_instance', 'naive_baseline', 'model_results', 'config']
    
    for key in required_keys:
        if key not in test_analysis:
            logger.error(f"❌ Missing required key in test analysis: {key}")
            return False
    
    instance_keys = ['coords', 'demands', 'capacity', 'num_customers']
    for key in instance_keys:
        if key not in test_analysis['test_instance']:
            logger.error(f"❌ Missing test instance data: {key}")
            return False
    
    if not test_analysis['model_results']:
        logger.error("❌ No model results found in test analysis")
        return False
    
    logger.info(f"✅ Test analysis validation passed")
    return True

def create_config_from_analysis(test_analysis, logger):
    """Create a config dict compatible with visualization functions"""
    config = test_analysis.get('config', {})
    
    # Extract essential config from test instance if missing
    if 'num_customers' not in config:
        config['num_customers'] = test_analysis['test_instance']['num_customers']
    
    if 'capacity' not in config:
        config['capacity'] = test_analysis['test_instance']['capacity']
    
    # Set reasonable defaults for missing visualization parameters
    config.setdefault('coord_range', 100)  # Will be inferred from data
    config.setdefault('demand_range', (1, 10))  # Will be inferred from data
    
    return config

def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info("🎨 Standalone Route Visualization Generator")
    logger.info("=" * 50)
    
    # Load test results
    test_analysis = load_test_results(args.results_path, logger)
    
    # Validate data
    if not validate_test_analysis(test_analysis, logger):
        sys.exit(1)
    
    # Create config
    config = create_config_from_analysis(test_analysis, logger)
    
    # Determine output directories
    if args.output_dir:
        plots_dir = os.path.join(args.output_dir, 'plots')
        analysis_dir = args.output_dir
    else:
        plots_dir = f"results/{args.scale}/plots"
        analysis_dir = f"results/{args.scale}/analysis"
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    logger.info(f"📁 Output directories:")
    logger.info(f"   Plots: {plots_dir}")
    logger.info(f"   Analysis: {analysis_dir}")
    
    # Check if visualizations already exist
    html_path = os.path.join(analysis_dir, 'test_instance_analysis.html')
    if os.path.exists(html_path) and not args.force_regenerate:
        logger.warning(f"⚠️  Visualizations already exist at {html_path}")
        logger.warning("   Use --force_regenerate to overwrite existing files")
        
        response = input("Continue anyway? [y/N]: ").lower()
        if response != 'y':
            logger.info("Aborted by user")
            sys.exit(0)
    
    # Generate visualizations
    try:
        from visualize_test_routes import plot_test_instance_routes, create_interactive_route_analysis
        
        success_count = 0
        
        if not args.html_only:
            logger.info("🖼️  Generating individual route plots...")
            plot_test_instance_routes(test_analysis, config, logger, plots_dir)
            success_count += 1
            logger.info(f"   ✅ Route plots saved to {plots_dir}")
        
        if not args.plots_only:
            logger.info("📄 Generating interactive HTML report...")
            create_interactive_route_analysis(test_analysis, config, analysis_dir)
            success_count += 1
            logger.info(f"   ✅ HTML report saved to {html_path}")
        
        logger.info(f"\n🎉 Successfully generated {success_count} visualization(s)!")
        
        # Show summary of what was created
        if not args.html_only:
            # Count plot files
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"   📊 Created {len(plot_files)} individual plot files")
        
        if not args.plots_only:
            # Show HTML report size
            if os.path.exists(html_path):
                size_mb = os.path.getsize(html_path) / (1024 * 1024)
                logger.info(f"   📄 HTML report size: {size_mb:.2f} MB")
        
        # Show model comparison summary
        logger.info(f"\n📈 Model Comparison Summary (from loaded results):")
        logger.info("-" * 60)
        
        naive_cost = test_analysis['naive_baseline']['cost_per_customer']
        logger.info(f"{'Model':<20} {'Greedy/Cust':<12} {'Sample/Cust':<12} {'Best Impr':<10}")
        logger.info("-" * 60)
        
        for model_name, results in test_analysis['model_results'].items():
            greedy_cost = results['greedy_cost_per_customer']
            sample_cost = results['sample_cost_per_customer']
            best_improvement = max(results['greedy_improvement'], results['sample_improvement'])
            
            logger.info(f"{model_name:<20} {greedy_cost:<12.3f} {sample_cost:<12.3f} {best_improvement:<10.1f}%")
        
        logger.info(f"{'Naive Baseline':<20} {naive_cost:<12.3f} {naive_cost:<12.3f} {'0.0':<10}%")
        logger.info("-" * 60)
    
    except ImportError as e:
        logger.error(f"❌ Failed to import visualization module: {e}")
        logger.error("   Make sure visualize_test_routes.py is in the current directory")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Failed to generate visualizations: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
