#!/usr/bin/env python3
"""
Smart Gradient-Based Hyperparameter Search for CVRP Models

Progressive search strategy:
1. GAT+RL: Target < 0.451 (excellent if achieved)
2. GT+RL: Target to beat GAT+RL's best result  
3. DGT+RL: Target to beat GT+RL's best result

Uses intelligent search with statistical significance testing and adaptive strategies.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import time
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
import random

class SmartHyperparameterSearch:
    def __init__(self, config_path: str = "configs/smart_search.yaml"):
        # Load base configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Search targets and thresholds
        self.absolute_minimum = 0.429  # Theoretical minimum
        self.excellent_threshold = 0.451  # 5% above minimum (0.429 + 5%)
        
        # Model progression order
        self.models = ['GAT+RL', 'GT+RL', 'DGT+RL']
        self.model_targets = {}  # Will be set dynamically
        
        # Search parameters bounds
        self.param_bounds = {
            'embedding_dim': (64, 1024),      # Hidden dimension range
            'n_layers': (2, 16),              # Number of layers  
            'n_heads': (2, 32),               # Attention heads (must be factor of embedding_dim)
            'learning_rate': (1e-5, 1e-2),   # Learning rate range
            'dropout': (0.0, 0.5),            # Dropout range
            'batch_size': (128, 2048),        # Batch size range
            'temp_start': (1.0, 10.0),        # Temperature start
            'temp_min': (0.01, 0.5),          # Temperature minimum
            'temp_decay': (0.1, 0.5)          # Temperature decay rate (maps to temp_adaptation_rate)
        }
        
        # Search state
        self.results = []
        self.best_configs = {}
        self.search_history = {}
        
        # Statistical testing parameters
        self.min_repeats = 3  # Minimum repetitions for statistical significance
        self.confidence_level = 0.95
        self.improvement_threshold = 0.01  # 1% improvement to be considered significant
        
        # Adaptive search parameters
        self.max_iterations = 30  # Maximum iterations per model
        self.stagnation_limit = 5   # Stop if no improvement for N iterations
        self.jump_probability = 0.3  # Probability to jump to random region when stagnating
        
    def log(self, message: str, level: str = "INFO"):
        """Logging with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def generate_initial_config(self, model: str) -> Dict:
        """Generate initial configuration based on temperature experiment results."""
        if model == 'GAT+RL':
            # Start with best configuration from temperature experiments
            return {
                'embedding_dim': 128,
                'n_layers': 4,
                'n_heads': 8,
                'learning_rate': 1e-3,
                'dropout': 0.1,
                'batch_size': 512,
                'temp_start': 5.0,  # Aggressive regime was best
                'temp_min': 0.02,
                'temp_decay': 0.15
            }
        else:
            # For GT+RL and DGT+RL, start with best GAT+RL config if available
            if 'GAT+RL' in self.best_configs:
                base_config = self.best_configs['GAT+RL'].copy()
                # Slight modifications for different architectures
                if model == 'GT+RL':
                    base_config['n_heads'] = min(16, base_config['embedding_dim'] // 8)
                elif model == 'DGT+RL':
                    base_config['n_heads'] = min(12, base_config['embedding_dim'] // 8)
                return base_config
            else:
                # Fallback to reasonable defaults
                return {
                    'embedding_dim': 256,
                    'n_layers': 6,
                    'n_heads': 8,
                    'learning_rate': 5e-4,
                    'dropout': 0.1,
                    'batch_size': 512,
                    'temp_start': 5.0,
                    'temp_min': 0.02,
                    'temp_decay': 0.15
                }
    
    def validate_config(self, config: Dict) -> Dict:
        """Validate and fix configuration constraints."""
        config = config.copy()
        
        # Ensure n_heads is a factor of embedding_dim and within bounds
        embedding_dim = int(config['embedding_dim'])
        n_heads = int(config['n_heads'])
        
        # Find valid number of heads (divisor of embedding_dim)
        possible_heads = [h for h in range(2, min(33, embedding_dim // 2 + 1)) 
                         if embedding_dim % h == 0]
        if possible_heads:
            config['n_heads'] = min(possible_heads, key=lambda x: abs(x - n_heads))
        else:
            config['n_heads'] = 8  # Fallback
        
        # Ensure integer parameters are integers
        for param in ['embedding_dim', 'n_layers', 'n_heads', 'batch_size']:
            config[param] = int(config[param])
        
        # Ensure batch_size is power of 2 for efficiency
        batch_size = config['batch_size']
        config['batch_size'] = 2 ** round(np.log2(batch_size))
        
        # Clamp all parameters to bounds
        for param, (min_val, max_val) in self.param_bounds.items():
            if param in config:
                config[param] = max(min_val, min(max_val, config[param]))
        
        return config
    
    def run_experiment(self, model: str, config: Dict, experiment_id: str) -> float:
        """Run single experiment and return validation cost."""
        validated_config = self.validate_config(config)
        
        command = [
            'python', 'run_experimental_training.py',
            '--models', model,
            '--config', 'configs/smart_search.yaml'
        ]
        
        # Add hyperparameters as command line arguments
        for param, value in validated_config.items():
            command.extend([f'--{param}', str(value)])
        
        start_time = time.time()
        self.log(f"🔬 Running {model} experiment {experiment_id}")
        # Print concise config info
        concise_config = f"dim={validated_config['embedding_dim']}, heads={validated_config['n_heads']}, lr={validated_config['learning_rate']:.1e}, temp={validated_config['temp_start']:.1f}"
        self.log(f"   {concise_config}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)  # Don't raise on error
            elapsed_time = time.time() - start_time
            
            # Check if process failed
            if result.returncode != 0:
                # Extract error details
                error_msg = "Unknown error"
                if result.stderr:
                    # Look for common error patterns
                    stderr_lines = result.stderr.strip().split('\n')
                    for line in reversed(stderr_lines[-10:]):  # Check last 10 lines
                        if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                            error_msg = line.strip()
                            break
                    if error_msg == "Unknown error":
                        error_msg = stderr_lines[-1] if stderr_lines else "No error message"
                elif result.stdout:
                    # Check stdout for error messages
                    stdout_lines = result.stdout.strip().split('\n')
                    for line in reversed(stdout_lines[-10:]):
                        if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                            error_msg = line.strip()
                            break
                
                self.log(f"❌ FAILED ({elapsed_time:.1f}s): {error_msg}", "ERROR")
                return float('inf')
            
            # Parse validation cost from output
            val_cost = float('nan')
            output_lines = result.stdout.strip().split('\n')
            
            # Try multiple patterns to find validation cost
            for line in reversed(output_lines):
                line_lower = line.lower()
                if 'final validation cost' in line_lower or 'validation cost:' in line_lower:
                    try:
                        # Extract number from line
                        parts = line.split(':')
                        if len(parts) > 1:
                            val_cost = float(parts[-1].strip())
                            break
                    except ValueError:
                        continue
                elif 'cost' in line_lower and any(word in line_lower for word in ['final', 'best', 'validation']):
                    try:
                        # Try to extract floating point number from line
                        import re
                        numbers = re.findall(r'\d+\.\d+', line)
                        if numbers:
                            val_cost = float(numbers[-1])  # Take last number found
                            break
                    except ValueError:
                        continue
            
            if np.isnan(val_cost):
                self.log(f"❌ FAILED: No cost found in output", "ERROR")
                # Print last few lines of output for debugging
                if len(output_lines) > 0:
                    self.log(f"   Last output lines ({len(output_lines)} total):", "DEBUG")
                    for line in output_lines[-10:]:
                        if line.strip():
                            self.log(f"   '{line.strip()}'", "DEBUG")
                else:
                    self.log(f"   NO OUTPUT LINES FOUND", "DEBUG")
                
                # Also print stderr if available
                if result.stderr:
                    self.log(f"   STDERR:", "DEBUG")
                    for line in result.stderr.strip().split('\n')[-5:]:
                        if line.strip():
                            self.log(f"   '{line.strip()}'", "DEBUG")
                
                return float('inf')
            
            # Log result with breakthrough detection
            breakthrough = "🎯 BREAKTHROUGH!" if val_cost < self.excellent_threshold else ""
            # Print concise result: cost/customer followed by key params
            cost_per_customer = val_cost * 20  # Assuming 20 customers
            self.log(f"✅ {breakthrough} {cost_per_customer:.1f}/cust (val_cost={val_cost:.4f}) | {concise_config} | {elapsed_time:.1f}s")
            
            return val_cost
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.log(f"❌ EXCEPTION ({elapsed_time:.1f}s): {str(e)}", "ERROR")
            return float('inf')
    
    def run_repeated_experiment(self, model: str, config: Dict, n_repeats: int = 3) -> Tuple[float, float]:
        """Run experiment multiple times for statistical significance."""
        costs = []
        for i in range(n_repeats):
            experiment_id = f"{len(self.results)+1}.{i+1}"
            cost = self.run_experiment(model, config, experiment_id)
            if not np.isinf(cost):
                costs.append(cost)
        
        if not costs:
            return float('inf'), float('inf')
        
        mean_cost = np.mean(costs)
        std_cost = np.std(costs) if len(costs) > 1 else 0.0
        
        self.log(f"📊 Repeated experiment result: {mean_cost:.4f} ± {std_cost:.4f} (n={len(costs)})")
        return mean_cost, std_cost
    
    def is_significant_improvement(self, new_cost: float, new_std: float, 
                                 current_best: float, current_std: float) -> bool:
        """Test if improvement is statistically significant."""
        if new_cost >= current_best - self.improvement_threshold:
            return False
        
        # If we have standard deviations, perform t-test
        if new_std > 0 and current_std > 0:
            # Simple t-test approximation
            pooled_std = np.sqrt((new_std**2 + current_std**2) / 2)
            if pooled_std > 0:
                t_stat = abs(new_cost - current_best) / pooled_std
                # Rough significance test (assuming df=4)
                return t_stat > 2.0  # Approximately 95% confidence
        
        # Fallback to simple threshold
        return (current_best - new_cost) > self.improvement_threshold
    
    def generate_neighbor_config(self, config: Dict, step_size: float = 0.2) -> Dict:
        """Generate neighbor configuration using gradient-like step."""
        neighbor = config.copy()
        
        # Add random perturbations to each parameter
        for param, (min_val, max_val) in self.param_bounds.items():
            if param in neighbor:
                current = neighbor[param]
                
                if param in ['embedding_dim', 'n_layers', 'n_heads', 'batch_size']:
                    # Integer parameters: small discrete steps
                    delta = random.choice([-2, -1, 0, 1, 2])
                    neighbor[param] = max(min_val, min(max_val, current + delta))
                else:
                    # Float parameters: proportional steps
                    range_size = max_val - min_val
                    delta = np.random.normal(0, step_size * range_size)
                    neighbor[param] = max(min_val, min(max_val, current + delta))
        
        return self.validate_config(neighbor)
    
    def generate_random_config(self) -> Dict:
        """Generate completely random configuration for exploration jumps."""
        config = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            if param in ['embedding_dim', 'n_layers', 'n_heads', 'batch_size']:
                # Integer parameters
                config[param] = random.randint(int(min_val), int(max_val))
            else:
                # Float parameters
                if param == 'learning_rate':
                    # Log-uniform for learning rate
                    config[param] = 10 ** random.uniform(np.log10(min_val), np.log10(max_val))
                else:
                    config[param] = random.uniform(min_val, max_val)
        
        return self.validate_config(config)
    
    def search_model(self, model: str, target_cost: float) -> Dict:
        """Smart search for optimal hyperparameters for a single model."""
        self.log(f"\n🚀 Starting smart search for {model}")
        self.log(f"🎯 Target: < {target_cost:.4f}")
        
        # Initialize search
        current_config = self.generate_initial_config(model)
        current_cost, current_std = self.run_repeated_experiment(model, current_config, self.min_repeats)
        
        best_config = current_config.copy()
        best_cost = current_cost
        best_std = current_std
        
        # Search state tracking
        iterations_without_improvement = 0
        search_history = []
        
        self.log(f"📍 Starting point: {best_cost:.4f} ± {best_std:.4f}")
        
        for iteration in range(self.max_iterations):
            self.log(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Decide search strategy
            if iterations_without_improvement >= self.stagnation_limit or random.random() < self.jump_probability:
                # Jump to random region to escape local minima
                self.log("🦘 Jumping to random region to escape stagnation")
                candidate_config = self.generate_random_config()
                step_size = 0.5  # Large step for exploration
            else:
                # Generate neighbor configuration
                step_size = 0.2 * (1 + iterations_without_improvement * 0.1)  # Adaptive step size
                candidate_config = self.generate_neighbor_config(best_config, step_size)
            
            # Test candidate configuration
            candidate_cost, candidate_std = self.run_repeated_experiment(model, candidate_config, self.min_repeats)
            
            # Record history
            search_history.append({
                'iteration': iteration + 1,
                'config': candidate_config.copy(),
                'cost': candidate_cost,
                'std': candidate_std,
                'is_best': False
            })
            
            # Check for improvement
            if candidate_cost < best_cost and self.is_significant_improvement(
                candidate_cost, candidate_std, best_cost, best_std):
                
                improvement = ((best_cost - candidate_cost) / best_cost) * 100
                self.log(f"🎉 Significant improvement: {candidate_cost:.4f} (↓{improvement:.1f}%)")
                
                best_config = candidate_config.copy()
                best_cost = candidate_cost
                best_std = candidate_std
                iterations_without_improvement = 0
                search_history[-1]['is_best'] = True
                
                # Check if we reached the target
                if best_cost < target_cost:
                    self.log(f"🏆 TARGET ACHIEVED! {best_cost:.4f} < {target_cost:.4f}")
                    break
                    
            else:
                iterations_without_improvement += 1
                improvement = ((best_cost - candidate_cost) / best_cost) * 100
                self.log(f"📈 No significant improvement: {candidate_cost:.4f} (Δ{improvement:+.1f}%)")
            
            # Early stopping conditions
            if iterations_without_improvement >= self.stagnation_limit and iteration > 10:
                self.log(f"⏹️ Stopping due to stagnation after {iterations_without_improvement} iterations")
                break
            
            if best_cost < self.absolute_minimum * 1.02:  # Within 2% of theoretical minimum
                self.log(f"⭐ Near-optimal performance achieved: {best_cost:.4f}")
                break
        
        # Store results
        self.best_configs[model] = best_config
        self.search_history[model] = search_history
        
        # Performance evaluation
        if best_cost < self.excellent_threshold:
            status = "🌟 EXCELLENT"
        elif best_cost < target_cost:
            status = "✅ TARGET ACHIEVED"
        elif best_cost < 0.55:
            status = "👍 GOOD"
        else:
            status = "❌ POOR"
        
        self.log(f"\n📋 {model} SEARCH COMPLETE:")
        self.log(f"   Best cost: {best_cost:.4f} ± {best_std:.4f}")
        self.log(f"   Target: {target_cost:.4f} {status}")
        self.log(f"   Config: {best_config}")
        self.log(f"   Iterations: {len(search_history)}")
        
        return {
            'model': model,
            'best_config': best_config,
            'best_cost': best_cost,
            'best_std': best_std,
            'target_cost': target_cost,
            'search_history': search_history,
            'status': status
        }
    
    def save_results(self):
        """Save search results to files."""
        results_dir = 'results/smart_hyperparameter_search'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"smart_search_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                'search_results': self.results,
                'best_configs': self.best_configs,
                'search_history': self.search_history,
                'targets': self.model_targets,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        self.log(f"💾 Results saved to: {results_file}")
        
        # Create summary CSV
        summary_data = []
        for result in self.results:
            summary_data.append({
                'model': result['model'],
                'best_cost': result['best_cost'],
                'best_std': result['best_std'],
                'target_cost': result['target_cost'],
                'status': result['status'],
                'iterations': len(result['search_history']),
                'embedding_dim': result['best_config']['embedding_dim'],
                'n_layers': result['best_config']['n_layers'],
                'n_heads': result['best_config']['n_heads'],
                'learning_rate': result['best_config']['learning_rate'],
                'dropout': result['best_config']['dropout']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(results_dir, f"smart_search_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        self.log(f"📊 Summary saved to: {summary_file}")
    
    def create_plots(self):
        """Create visualization plots of the search process."""
        if not self.results:
            return
        
        results_dir = 'results/smart_hyperparameter_search'
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create search progress plot
        fig, axes = plt.subplots(len(self.results), 1, figsize=(12, 4*len(self.results)))
        if len(self.results) == 1:
            axes = [axes]
        
        for i, result in enumerate(self.results):
            ax = axes[i]
            model = result['model']
            history = result['search_history']
            
            iterations = [h['iteration'] for h in history]
            costs = [h['cost'] for h in history]
            best_points = [h for h in history if h['is_best']]
            
            # Plot all points
            ax.scatter(iterations, costs, alpha=0.6, label='Experiments')
            
            # Highlight best points
            if best_points:
                best_iterations = [h['iteration'] for h in best_points]
                best_costs = [h['cost'] for h in best_points]
                ax.scatter(best_iterations, best_costs, color='red', s=100, label='Improvements', zorder=5)
                
                # Connect best points
                ax.plot(best_iterations, best_costs, 'r--', alpha=0.7, linewidth=2)
            
            # Add target line
            target = result['target_cost']
            ax.axhline(y=target, color='green', linestyle='--', alpha=0.7, label=f'Target: {target:.3f}')
            
            # Add excellent threshold
            ax.axhline(y=self.excellent_threshold, color='gold', linestyle='--', alpha=0.7, 
                      label=f'Excellent: {self.excellent_threshold:.3f}')
            
            ax.set_title(f'{model} Hyperparameter Search Progress\nBest: {result["best_cost"]:.4f}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Validation Cost')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, 'search_progress.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"📈 Search progress plot saved: {plot_file}")
    
    def run_progressive_search(self):
        """Run the complete progressive search across all models."""
        self.log("🎯 SMART PROGRESSIVE HYPERPARAMETER SEARCH")
        self.log("=" * 60)
        self.log(f"Goals:")
        self.log(f"  1. GAT+RL: Achieve < {self.excellent_threshold:.3f} (excellent threshold)")
        self.log(f"  2. GT+RL: Beat GAT+RL's best result")
        self.log(f"  3. DGT+RL: Beat GT+RL's best result")
        self.log(f"  Absolute minimum: {self.absolute_minimum:.3f}")
        
        # Progressive search through models
        for i, model in enumerate(self.models):
            # Set target based on previous results
            if model == 'GAT+RL':
                target = self.excellent_threshold
            else:
                # Target to beat previous model's best
                prev_model = self.models[i-1]
                if prev_model in self.best_configs:
                    prev_best = min(r['best_cost'] for r in self.results if r['model'] == prev_model)
                    target = prev_best - 0.01  # Beat by at least 1%
                else:
                    target = 0.55  # Fallback
            
            self.model_targets[model] = target
            
            # Run search for this model
            result = self.search_model(model, target)
            self.results.append(result)
            
            # Early termination if we achieve excellent performance
            if result['best_cost'] < self.absolute_minimum * 1.05:
                self.log(f"🏆 Near-theoretical minimum achieved with {model}! Stopping search.")
                break
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("🏁 PROGRESSIVE SEARCH COMPLETE")
        self.log("="*60)
        
        for result in self.results:
            model = result['model']
            cost = result['best_cost']
            status = result['status']
            self.log(f"{model:12}: {cost:.4f} {status}")
        
        # Architecture advancement analysis
        if len(self.results) > 1:
            self.log(f"\n🚀 ARCHITECTURE ADVANCEMENT:")
            gat_cost = self.results[0]['best_cost']
            for i in range(1, len(self.results)):
                current_cost = self.results[i]['best_cost']
                current_model = self.results[i]['model']
                improvement = ((gat_cost - current_cost) / gat_cost) * 100
                
                if current_cost < gat_cost:
                    self.log(f"   {current_model} vs GAT+RL: ↓{improvement:.1f}% BETTER")
                else:
                    self.log(f"   {current_model} vs GAT+RL: ↑{-improvement:.1f}% WORSE")
        
        # Save results and create plots
        self.save_results()
        self.create_plots()

def main():
    searcher = SmartHyperparameterSearch()
    searcher.run_progressive_search()

if __name__ == '__main__':
    main()
