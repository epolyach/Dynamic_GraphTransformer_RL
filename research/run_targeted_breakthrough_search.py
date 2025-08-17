#!/usr/bin/env python3
"""
Targeted Breakthrough Search for <0.5 Performance

Uses the working small.yaml config as base and systematically tests 
promising hyperparameter combinations to achieve breakthrough performance.
"""

import os
import subprocess
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class TargetedBreakthroughSearch:
    def __init__(self):
        self.target_cost = 0.45  # Breakthrough target
        self.base_config = "configs/small.yaml"
        self.models = ['GAT+RL', 'GT+RL', 'DGT+RL']
        self.results = []
        
        # Focused search configurations based on RL literature and temperature results
        self.promising_configs = [
            # Start with known good configurations from temperature experiments
            {
                'name': 'baseline_aggressive',
                'embedding_dim': 128, 'n_layers': 4, 'n_heads': 8, 
                'learning_rate': 1e-3, 'dropout': 0.1,
                'temp_start': 5.0, 'temp_min': 0.02, 'temp_decay': 0.15
            },
            # Larger capacity configurations
            {
                'name': 'large_capacity_1',
                'embedding_dim': 256, 'n_layers': 6, 'n_heads': 8, 
                'learning_rate': 5e-4, 'dropout': 0.1,
                'temp_start': 5.0, 'temp_min': 0.02, 'temp_decay': 0.15
            },
            {
                'name': 'large_capacity_2',
                'embedding_dim': 256, 'n_layers': 8, 'n_heads': 16, 
                'learning_rate': 5e-4, 'dropout': 0.15,
                'temp_start': 7.0, 'temp_min': 0.02, 'temp_decay': 0.15
            },
            # Deep network configurations
            {
                'name': 'deep_network_1',
                'embedding_dim': 512, 'n_layers': 8, 'n_heads': 16, 
                'learning_rate': 1e-4, 'dropout': 0.2,
                'temp_start': 5.0, 'temp_min': 0.02, 'temp_decay': 0.15
            },
            {
                'name': 'deep_network_2',
                'embedding_dim': 512, 'n_layers': 12, 'n_heads': 32, 
                'learning_rate': 1e-4, 'dropout': 0.1,
                'temp_start': 7.0, 'temp_min': 0.02, 'temp_decay': 0.2
            },
            # Very high capacity for breakthrough attempt
            {
                'name': 'breakthrough_attempt',
                'embedding_dim': 1024, 'n_layers': 8, 'n_heads': 32, 
                'learning_rate': 5e-5, 'dropout': 0.15,
                'temp_start': 7.0, 'temp_min': 0.01, 'temp_decay': 0.1
            },
            # Alternative learning rates
            {
                'name': 'high_lr',
                'embedding_dim': 256, 'n_layers': 6, 'n_heads': 8, 
                'learning_rate': 2e-3, 'dropout': 0.1,
                'temp_start': 5.0, 'temp_min': 0.02, 'temp_decay': 0.15
            },
            {
                'name': 'low_lr_deep',
                'embedding_dim': 512, 'n_layers': 6, 'n_heads': 16, 
                'learning_rate': 1e-5, 'dropout': 0.05,
                'temp_start': 5.0, 'temp_min': 0.02, 'temp_decay': 0.15
            }
        ]
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_experiment(self, model: str, config: Dict, config_name: str) -> float:
        """Run single experiment and return validation cost."""
        
        # Build command using working small.yaml as base
        command = [
            'python', 'run_experimental_training.py',
            '--models', model,
            '--config', self.base_config,
            '--embedding_dim', str(config['embedding_dim']),
            '--n_layers', str(config['n_layers']),
            '--n_heads', str(config['n_heads']),
            '--learning_rate', str(config['learning_rate']),
            '--dropout', str(config['dropout']),
            '--temp_start', str(config['temp_start']),
            '--temp_min', str(config['temp_min']),
            '--temp_decay', str(config['temp_decay'])
        ]
        
        start_time = time.time()
        self.log(f"🔬 Running {model} with {config_name}")
        self.log(f"   Config: {config}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            # Parse validation cost from output
            val_cost = float('nan')
            for line in reversed(result.stdout.strip().split('\n')):
                if 'final validation cost' in line:
                    val_cost = float(line.split(':')[1].strip())
                    break
            
            if np.isnan(val_cost):
                self.log(f"❌ Failed to parse validation cost")
                return float('inf')
            
            # Check for breakthrough
            if val_cost < self.target_cost:
                breakthrough_msg = f"🎯 BREAKTHROUGH! {val_cost:.4f} < {self.target_cost:.3f}"
                self.log(f"✅ {breakthrough_msg} ({elapsed_time:.1f}s)")
            else:
                self.log(f"✅ Cost: {val_cost:.4f} ({elapsed_time:.1f}s)")
            
            return val_cost
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            self.log(f"❌ Experiment failed ({elapsed_time:.1f}s)")
            if e.stderr:
                self.log(f"   Error: {e.stderr[:200]}...")
            return float('inf')
    
    def run_systematic_search(self):
        """Run systematic search through all promising configurations."""
        self.log("🎯 TARGETED BREAKTHROUGH SEARCH")
        self.log("=" * 60)
        self.log(f"Target: < {self.target_cost:.3f}")
        self.log(f"Models: {self.models}")
        self.log(f"Configurations: {len(self.promising_configs)}")
        self.log(f"Total experiments: {len(self.models) * len(self.promising_configs)}")
        
        breakthrough_found = False
        experiment_count = 0
        
        for config in self.promising_configs:
            config_name = config.pop('name')  # Remove name from config dict
            
            self.log(f"\n📋 Testing configuration: {config_name}")
            
            for model in self.models:
                experiment_count += 1
                val_cost = self.run_experiment(model, config, config_name)
                
                # Record result
                result = {
                    'experiment_id': experiment_count,
                    'model': model,
                    'config_name': config_name,
                    'config': config.copy(),
                    'validation_cost': val_cost,
                    'timestamp': datetime.now().isoformat(),
                    'breakthrough': val_cost < self.target_cost
                }
                self.results.append(result)
                
                # Check for breakthrough
                if val_cost < self.target_cost:
                    breakthrough_found = True
                    self.log(f"🚀 FIRST BREAKTHROUGH: {model} with {config_name}!")
                
                # Save incremental results
                self.save_results()
        
        # Final analysis
        self.analyze_results(breakthrough_found)
    
    def save_results(self):
        """Save results to JSON file."""
        results_dir = 'results/targeted_breakthrough_search'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'breakthrough_search_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump({
                'search_results': self.results,
                'target_cost': self.target_cost,
                'timestamp': timestamp,
                'total_experiments': len(self.results)
            }, f, indent=2, default=str)
    
    def analyze_results(self, breakthrough_found: bool):
        """Analyze and report final results."""
        self.log("\n" + "=" * 60)
        self.log("🏁 TARGETED BREAKTHROUGH SEARCH COMPLETE")
        self.log("=" * 60)
        
        if not self.results:
            self.log("❌ No successful experiments completed.")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if not np.isinf(r['validation_cost'])]
        
        if not successful_results:
            self.log("❌ No successful experiments completed.")
            return
        
        # Find best results
        best_results = sorted(successful_results, key=lambda x: x['validation_cost'])
        
        self.log(f"📊 SUMMARY:")
        self.log(f"   Total experiments: {len(self.results)}")
        self.log(f"   Successful: {len(successful_results)}")
        self.log(f"   Breakthrough target: < {self.target_cost:.3f}")
        
        # Count breakthroughs
        breakthroughs = [r for r in successful_results if r['breakthrough']]
        self.log(f"   Breakthroughs achieved: {len(breakthroughs)}")
        
        if breakthroughs:
            self.log(f"\n🎯 BREAKTHROUGH RESULTS:")
            self.log("-" * 80)
            for i, result in enumerate(sorted(breakthroughs, key=lambda x: x['validation_cost']), 1):
                self.log(f"{i:2d}. {result['model']:12} | {result['config_name']:20} | Cost: {result['validation_cost']:.4f}")
                self.log(f"     Config: emb={result['config']['embedding_dim']}, layers={result['config']['n_layers']}, "
                         f"heads={result['config']['n_heads']}, lr={result['config']['learning_rate']:.1e}")
        
        self.log(f"\n🏆 TOP 10 RESULTS (regardless of breakthrough):")
        self.log("-" * 80)
        for i, result in enumerate(best_results[:10], 1):
            breakthrough_marker = "🎯" if result['breakthrough'] else "  "
            self.log(f"{breakthrough_marker}{i:2d}. {result['model']:12} | {result['config_name']:20} | Cost: {result['validation_cost']:.4f}")
        
        # Architecture comparison
        self.log(f"\n📈 BEST PERFORMANCE BY MODEL:")
        self.log("-" * 50)
        for model in self.models:
            model_results = [r for r in successful_results if r['model'] == model]
            if model_results:
                best_model_result = min(model_results, key=lambda x: x['validation_cost'])
                breakthrough_status = "🎯" if best_model_result['breakthrough'] else "  "
                self.log(f"{breakthrough_status}{model:12}: {best_model_result['validation_cost']:.4f} "
                         f"({best_model_result['config_name']})")
        
        if breakthrough_found:
            self.log(f"\n🌟 SUCCESS: Breakthrough performance achieved!")
        else:
            best_cost = best_results[0]['validation_cost']
            gap = best_cost - self.target_cost
            self.log(f"\n📊 Best achieved: {best_cost:.4f} (gap: +{gap:.4f} from target {self.target_cost:.3f})")
        
        # Save final results
        self.save_results()
        self.log(f"\n💾 Results saved to: results/targeted_breakthrough_search/")

def main():
    searcher = TargetedBreakthroughSearch()
    searcher.run_systematic_search()

if __name__ == '__main__':
    main()
