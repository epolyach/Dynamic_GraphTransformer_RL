#!/usr/bin/env python3
"""
Sequential hyperparameter optimization for CVRP models.
Optimizes GAT+RL first, then GT+RL starting from GAT's best params, 
then DGT+RL starting from GT's best params.
Goal: achieve DGT < GT < GAT validation costs.
"""

import os
import sys
import json
import yaml
import subprocess
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei


class SequentialModelOptimizer:
    def __init__(self, base_config_path: str = "configs/small.yaml"):
        self.base_config_path = base_config_path
        self.base_config = self.load_base_config()
        self.results = {}
        self.best_params_chain = {}
        self.step_counter = 0
        
        # Define hyperparameter search space - optimized for small instances (20 customers)
        # Includes parameter count control (5K to 300K parameters)
        self.param_space = [
            # Model Size Control (5K to 300K parameters)
            Integer(30, 84, name='hidden_dim'),       # Adjusted range for target parameter count
            Integer(2, 3, name='n_layers'),           # Max 3 layers for 20 customers
            Integer(2, 6, name='n_heads'),            # Attention heads (min 2 for stability) 
            Real(0.0, 0.25, name='dropout'),          # Dropout regularization
            Real(1.0, 2.2, name='ff_multiplier'),     # Feedforward layer size multiplier (reduced max)
            
            # Training Parameters
            Integer(1024, 4096, name='num_instances'), # Training instances per epoch
            Integer(128, 512, name='batch_size'),      # Batch size
            Real(1e-4, 5e-3, name='learning_rate', prior='log-uniform'),
            Real(1e-5, 1e-3, name='weight_decay', prior='log-uniform'),
            
            # RL-Critical Temperature Parameters
            Real(1.5, 4.0, name='temp_start'),        # Initial exploration temperature
            Real(0.1, 0.5, name='temp_min'),          # Minimum temperature
            Real(0.05, 0.3, name='temp_adaptation_rate'), # Temperature adaptation
            
            # Advanced Training Parameters
            Integer(10, 30, name='early_stopping_patience'),
            Real(0.001, 0.02, name='entropy_coef'),   # Entropy regularization
            Real(0.5, 3.0, name='gradient_clip_norm'), # Gradient clipping
        ]
        
        self.param_names = [dim.name for dim in self.param_space]
        
        # Results storage
        self.log_file = f"sequential_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.results_file = f"sequential_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def load_base_config(self) -> Dict:
        """Load base configuration file."""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def log_message(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def params_to_dict(self, param_values: List[float]) -> Dict[str, Any]:
        """Convert parameter list to dictionary."""
        params = {}
        integer_params = ['hidden_dim', 'n_layers', 'n_heads', 'num_instances', 'batch_size', 'early_stopping_patience']
        
        for name, value in zip(self.param_names, param_values):
            if name in integer_params:
                params[name] = int(value)  # Explicit int conversion
            else:
                params[name] = float(value)  # Explicit float conversion
        return params
    
    def create_config_with_params(self, model_name: str, params: Dict[str, Any]) -> Dict:
        """Create configuration with given parameters."""
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Ensure all sections exist
        if 'model' not in config:
            config['model'] = {}
        if 'training' not in config:
            config['training'] = {}
        if 'training_advanced' not in config:
            config['training_advanced'] = {}
        
        # Update model-specific parameters
        config['model']['name'] = model_name
        config['model']['hidden_dim'] = params.get('hidden_dim', 64)
        config['model']['num_layers'] = params.get('n_layers', 3)
        config['model']['num_heads'] = params.get('n_heads', 8)
        config['model']['transformer_dropout'] = params.get('dropout', 0.1)
        config['model']['feedforward_multiplier'] = params.get('ff_multiplier', 2.0)
        
        # Update training parameters
        config['training']['num_instances'] = params.get('num_instances', 2048)
        config['training']['batch_size'] = params.get('batch_size', 256)
        config['training']['learning_rate'] = params.get('learning_rate', 1e-3)
        
        # Update advanced training parameters
        config['training_advanced']['weight_decay'] = params.get('weight_decay', 1e-4)
        config['training_advanced']['temp_start'] = params.get('temp_start', 2.0)
        config['training_advanced']['temp_min'] = params.get('temp_min', 0.2)
        config['training_advanced']['temp_adaptation_rate'] = params.get('temp_adaptation_rate', 0.1)
        config['training_advanced']['early_stopping_patience'] = params.get('early_stopping_patience', 15)
        config['training_advanced']['entropy_coef'] = params.get('entropy_coef', 0.01)
        config['training_advanced']['gradient_clip_norm'] = params.get('gradient_clip_norm', 2.0)
        
        return config
    
    def run_single_experiment(self, model_name: str, params: Dict[str, Any], verbose: bool = False) -> Tuple[float, str]:
        """Run a single training experiment and return validation cost and error info."""
        error_msg = ""
        try:
            # Create temporary config file
            temp_config_file = f"temp_config_{model_name}_{int(time.time())}.yaml"
            config = self.create_config_with_params(model_name, params)
            
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Run training
            cmd = [
                sys.executable, "run_experimental_training.py",
                "--config", temp_config_file,
                "--model", model_name
            ]
            
            if verbose:
                self.log_message(f"Running: {' '.join(cmd)}")
                self.log_message(f"Parameters: {params}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Clean up temp file
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            
            if result.returncode != 0:
                # Extract error from stderr
                stderr_lines = result.stderr.strip().split('\n')
                if 'AssertionError: embed_dim must be divisible by num_heads' in result.stderr:
                    error_msg = "embed_dim not divisible by heads"
                elif 'CUDA out of memory' in result.stderr:
                    error_msg = "GPU memory error"
                elif 'Traceback' in result.stderr:
                    # Get the actual error line
                    for line in stderr_lines:
                        if 'Error:' in line or 'Exception:' in line or 'AssertionError:' in line:
                            error_msg = line.split(':')[-1].strip()[:50]
                            break
                    if not error_msg:
                        error_msg = "training crashed"
                else:
                    error_msg = f"exit code {result.returncode}"
                
                if verbose:
                    self.log_message(f"Training failed: {error_msg}")
                    self.log_message(f"STDERR: {result.stderr}")
                return float('inf'), error_msg
            
            # Extract validation cost from output
            output = result.stdout
            validation_cost = self.extract_validation_cost(output)
            
            if validation_cost is None:
                error_msg = "no cost found in output"
                if verbose:
                    self.log_message(f"Could not extract validation cost from output")
                return float('inf'), error_msg
            
            if verbose:
                self.log_message(f"Validation cost: {validation_cost}")
            return validation_cost, ""
            
        except subprocess.TimeoutExpired:
            error_msg = "timeout (>1hr)"
            if verbose:
                self.log_message(f"Training timeout for {model_name}")
            return float('inf'), error_msg
        except Exception as e:
            error_msg = f"exception: {str(e)[:30]}"
            if verbose:
                self.log_message(f"Error in experiment: {str(e)}")
            return float('inf'), error_msg
    
    def extract_validation_cost(self, output: str) -> float:
        """Extract final validation cost from training output."""
        # Look for patterns like "Final validation cost: 0.xxxx"
        patterns = [
            r'Final validation cost:\s*([0-9.]+)',
            r'Best validation cost:\s*([0-9.]+)',
            r'Validation cost:\s*([0-9.]+)',
            r'Val cost:\s*([0-9.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        return None
    
    def optimize_single_model(self, model_name: str, starting_params: Dict[str, Any] = None, n_calls: int = 30) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters for a single model."""
        print(f"\n🎯 Optimizing {model_name}")
        print(f"📊 Will try {n_calls} different hyperparameter combinations")
        print(f"🎯 Target: cost per customer ≤ 0.45")
        print("─" * 60)
        
        # Track all results to handle failures
        all_results = []
        
        # Create objective function
        @use_named_args(self.param_space)
        def objective(**params):
            self.step_counter += 1
            # Convert numpy types to native Python types
            clean_params = {}
            integer_params = ['hidden_dim', 'n_layers', 'n_heads', 'num_instances', 'batch_size', 'early_stopping_patience']
            
            for k, v in params.items():
                if k in integer_params:
                    clean_params[k] = int(v)  # Force conversion to int
                else:
                    clean_params[k] = float(v)  # Force conversion to float
            
            # Ensure hidden_dim is divisible by n_heads
            hidden_dim = clean_params['hidden_dim']
            n_heads = clean_params['n_heads']
            if hidden_dim % n_heads != 0:
                clean_params['hidden_dim'] = ((hidden_dim // n_heads) + 1) * n_heads
            
            # Periodic verbose output (every 10 steps)
            verbose = (self.step_counter % 10 == 0)
            
            cost, error_msg = self.run_single_experiment(model_name, clean_params, verbose=verbose)
            # Cap infinite costs to a large but finite value
            if np.isinf(cost) or np.isnan(cost):
                cost = 10.0  # Much higher than expected good costs (typically 0.4-0.6)
            
            all_results.append((clean_params.copy(), cost))
            # Immediate output after each evaluation
            if cost != 10.0:  # Only print successful runs
                print(f"{model_name}: {cost:.3f} (h={clean_params['hidden_dim']}, heads={clean_params['n_heads']}, lr={clean_params['learning_rate']:.0e}, temp={clean_params['temp_start']:.1f})")
            else:
                print(f"{model_name}: FAILED - {error_msg} (h={clean_params['hidden_dim']}, heads={clean_params['n_heads']}, lr={clean_params['learning_rate']:.0e})")
            
            if verbose and cost != 10.0:
                print(f"Full parameters: {clean_params}")
            
            return cost
        
        # Set initial point if provided
        x0 = None
        if starting_params:
            x0 = [starting_params.get(name, self.param_space[i].low) for i, name in enumerate(self.param_names)]
        
        # Run Bayesian optimization
        self.log_message(f"Starting Bayesian optimization with {n_calls} calls")
        
        try:
            result = gp_minimize(
                func=objective,
                dimensions=self.param_space,
                n_calls=n_calls,
                n_initial_points=5 if x0 is None else 1,
                x0=x0,
                acq_func='EI',  # Expected Improvement
                random_state=42
            )
            
            # Extract best parameters and cost from optimization result
            best_params = self.params_to_dict(result.x)
            best_cost = result.fun
            
        except Exception as e:
            self.log_message(f"Bayesian optimization failed: {str(e)}")
            # Fallback: find best from all attempted results
            if all_results:
                best_result = min(all_results, key=lambda x: x[1])
                best_params = best_result[0]
                best_cost = best_result[1]
                self.log_message(f"Using best result from attempted configurations")
            else:
                # Ultimate fallback: return default parameters
                best_params = {
                    'embedding_dim': 128,
                    'n_layers': 3,
                    'n_heads': 8,
                    'learning_rate': 1e-3,
                    'dropout': 0.1,
                    'temperature': 1.0,
                    'weight_decay': 1e-5
                }
                best_cost = 10.0
                self.log_message(f"Using fallback default parameters")
        
        self.log_message(f"\nOptimization complete for {model_name}")
        self.log_message(f"Best validation cost: {best_cost}")
        self.log_message(f"Best parameters: {best_params}")
        
        return best_params, best_cost
    
    def run_sequential_optimization(self, models: List[str] = None, n_calls_per_model: int = 30):
        """Run sequential optimization across models."""
        if models is None:
            models = ['GAT+RL', 'GT+RL', 'DGT+RL']
        
        target_cost = 0.45  # Target cost per customer
        starting_params = None
        
        for i, model_name in enumerate(models):
            # Optimize current model
            best_params, best_cost = self.optimize_single_model(
                model_name, 
                starting_params=starting_params,
                n_calls=n_calls_per_model
            )
            
            # Store results
            self.results[model_name] = {
                'best_params': best_params,
                'best_cost': best_cost,
                'starting_params': starting_params
            }
            
            # Save incremental results
            self.save_results()
            
            # Check if goal achieved and decide on next action
            if best_cost <= target_cost:
                print(f"✅ {model_name} achieved target {target_cost} with cost {best_cost:.3f} → Switching to next model")
                reason = "goal achieved"
            elif best_cost == 10.0:  # All runs failed
                print(f"❌ {model_name} all attempts failed (cost {best_cost:.3f}) → Switching to next model")
                reason = "all runs failed"
            else:
                print(f"⚠️ {model_name} best cost {best_cost:.3f} (target {target_cost}) → Switching to next model")
                reason = "optimization completed"
            
            # Use current best as starting point for next model
            starting_params = best_params.copy()
            self.best_params_chain[model_name] = best_params
            
        self.log_message(f"\n{'='*60}")
        self.log_message("SEQUENTIAL OPTIMIZATION COMPLETE")
        self.log_message(f"{'='*60}")
        
        # Print final summary
        self.print_final_summary()
    
    def save_results(self):
        """Save current results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_final_summary(self):
        """Print final optimization summary."""
        self.log_message("\nFINAL RESULTS SUMMARY:")
        self.log_message("-" * 40)
        
        costs = []
        for model_name, result in self.results.items():
            cost = result['best_cost']
            costs.append(cost)
            self.log_message(f"{model_name:12s}: {cost:.6f}")
        
        if len(costs) >= 3:
            gat_cost, gt_cost, dgt_cost = costs[0], costs[1], costs[2]
            success = (dgt_cost < gt_cost < gat_cost)
            
            self.log_message(f"\nTarget achieved (DGT < GT < GAT): {success}")
            if success:
                self.log_message("✅ Sequential improvement achieved!")
            else:
                self.log_message("❌ Sequential improvement not achieved")
                
        self.log_message(f"\nResults saved to: {self.results_file}")
        self.log_message(f"Log saved to: {self.log_file}")


def main():
    """Main execution function."""
    optimizer = SequentialModelOptimizer()
    
    # Run sequential optimization
    models = ['GAT+RL', 'GT+RL', 'DGT+RL']
    n_calls = 8  # Number of optimization calls per model (reduced for testing)
    
    try:
        optimizer.run_sequential_optimization(models, n_calls)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        optimizer.save_results()
        optimizer.log_message("Optimization interrupted by user")
    except Exception as e:
        optimizer.log_message(f"Optimization failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
