#!/usr/bin/env python3
"""
Ablation Study Runner for Dynamic Graph Transformer CVRP
Runs comprehensive comparison between three model variants:
0. Baseline: Greedy Graph Transformer
1. Static RL: Graph Transformer + RL  
2. Dynamic RL: Full pipeline with dynamic updates
"""

import os
import sys
import time
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# TEMPORARILY DISABLED: Models moved to models_backup/ during cleanup
# from src.models.ablation_models import create_ablation_model
# from src.data.instance_creator.InstanceGenerator import InstanceGenerator
# from src.utils.RL.euclidean_cost_eval import euclidean_cost
import warnings
warnings.filterwarnings("ignore")

print("⚠️  ABLATION STUDY TEMPORARILY DISABLED")
print("   Models have been moved to src/models_backup/ during cleanup.")
print("   To re-enable, restore required models to src/models/ or run from models_backup.")
exit(1)


class AblationStudyRunner:
    """
    Main class for running ablation study experiments
    """
    
    def __init__(self, config_path: str):
        """Initialize runner with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.experiment_name = self.config['experiment']['name']
        self.variants = self.config['experiment']['variants']
        self.problem_sizes = self.config['experiment']['problem_sizes']
        self.num_test_instances = self.config['experiment']['num_test_instances']
        self.num_runs = self.config['experiment']['num_runs']
        
        # Setup directories
        self.results_dir = Path(self.config['logging']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = self._setup_device()
        
        # Results storage
        self.results = {}
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config['hardware']['device'] == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config['hardware']['device'])
    
    def generate_test_instances(self, problem_size: int, num_instances: int) -> List[Any]:
        """Generate test instances for evaluation"""
        print(f"Generating {num_instances} test instances of size {problem_size}...")
        
        instances = []
        generator = InstanceGenerator()
        
        for i in range(num_instances):
            # Generate instance
            instance = generator.generate_instance(
                num_nodes=problem_size,
                vehicle_capacity=self.config['problem']['vehicle_capacity'],
                coord_range=self.config['problem']['coordinate_range'],
                demand_range=self.config['problem']['demand_range']
            )
            instances.append(instance)
            
        return instances
    
    def evaluate_model(self, model: torch.nn.Module, instances: List[Any], 
                      variant: str, problem_size: int) -> Dict[str, List[float]]:
        """Evaluate model on test instances"""
        print(f"Evaluating {variant} on {problem_size}-node instances...")
        
        model.eval()
        results = {
            'solution_cost': [],
            'computation_time': [],
            'memory_usage': []
        }
        
        with torch.no_grad():
            for instance in instances:
                # Multiple runs per instance
                instance_costs = []
                instance_times = []
                
                for run in range(self.num_runs):
                    # Prepare data
                    data = self._prepare_data(instance)
                    
                    # Time the forward pass
                    start_time = time.time()
                    
                    # Get GPU memory before
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_before = torch.cuda.memory_allocated()
                    
                    # Model forward pass
                    n_steps = problem_size + 5  # Allow extra steps
                    actions, log_probs = model(data, n_steps=n_steps, greedy=True)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_after = torch.cuda.memory_allocated()
                        memory_used = (mem_after - mem_before) / 1024**2  # MB
                    else:
                        memory_used = 0
                    
                    end_time = time.time()
                    computation_time = end_time - start_time
                    
                    # Compute solution cost
                    cost = self._compute_solution_cost(actions, instance)
                    
                    instance_costs.append(cost)
                    instance_times.append(computation_time)
                
                # Store results (average across runs)
                results['solution_cost'].append(np.mean(instance_costs))
                results['computation_time'].append(np.mean(instance_times))
                results['memory_usage'].append(memory_used)  # Same for all runs
        
        return results
    
    def _prepare_data(self, instance: Dict[str, Any]) -> Any:
        """Prepare instance data for model input"""
        # This is a simplified version - you'll need to adapt based on your data format
        coordinates = torch.tensor(instance['coordinates'], dtype=torch.float32, device=self.device)
        demands = torch.tensor(instance['demands'], dtype=torch.float32, device=self.device)
        
        # Create mock data structure - adapt based on your actual data format
        from torch_geometric.data import Data
        
        # Create fully connected graph
        num_nodes = len(coordinates)
        edge_indices = []
        edge_attrs = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_indices.append([i, j])
                    dist = torch.norm(coordinates[i] - coordinates[j])
                    edge_attrs.append([dist])
        
        edge_index = torch.tensor(edge_indices, device=self.device).t().long()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=self.device)
        
        # Create data object
        data = Data(
            x=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=demands.unsqueeze(-1),
            capacity=torch.full((num_nodes,), instance['vehicle_capacity'], 
                              dtype=torch.float32, device=self.device),
            batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        )
        data.num_graphs = 1
        
        return data
    
    def _compute_solution_cost(self, actions: torch.Tensor, instance: Dict[str, Any]) -> float:
        """Compute total routing cost for solution"""
        # Extract route from actions
        route = actions[0, :, 0].cpu().numpy()  # First batch, all steps, single action
        
        # Remove invalid/repeated actions
        valid_route = []
        visited = set()
        
        for node in route:
            node = int(node)
            if node not in visited or node == 0:  # Allow multiple depot visits
                valid_route.append(node)
                if node != 0:
                    visited.add(node)
        
        # Ensure route ends at depot
        if valid_route[-1] != 0:
            valid_route.append(0)
        
        # Compute total distance
        total_cost = 0.0
        coordinates = instance['coordinates']
        
        for i in range(len(valid_route) - 1):
            current_node = valid_route[i]
            next_node = valid_route[i + 1]
            
            coord1 = coordinates[current_node]
            coord2 = coordinates[next_node]
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            total_cost += distance
        
        return total_cost
    
    def run_experiment(self):
        """Run complete ablation study"""
        print(f"Starting Ablation Study: {self.experiment_name}")
        print(f"Variants: {self.variants}")
        print(f"Problem sizes: {self.problem_sizes}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        # Initialize results storage
        for variant in self.variants:
            self.results[variant] = {}
            
        # Run experiments for each problem size
        for problem_size in self.problem_sizes:
            print(f"\\n=== Problem Size: {problem_size} nodes ===")
            
            # Generate test instances
            instances = self.generate_test_instances(problem_size, self.num_test_instances)
            
            # Test each variant
            for variant in self.variants:
                print(f"\\nTesting variant: {variant}")
                
                # Create model
                model = create_ablation_model(variant, self.config['model'])
                model.to(self.device)
                
                # Evaluate model
                variant_results = self.evaluate_model(model, instances, variant, problem_size)
                
                # Store results
                self.results[variant][problem_size] = variant_results
                
                # Print summary
                avg_cost = np.mean(variant_results['solution_cost'])
                avg_time = np.mean(variant_results['computation_time'])
                print(f"  Average cost: {avg_cost:.2f}")
                print(f"  Average time: {avg_time:.4f}s")
                
                # Clean up GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save and analyze results
        self.save_results()
        self.analyze_results()
        self.generate_plots()
        
        print(f"\\n=== Experiment Complete ===")
        print(f"Results saved to: {self.results_dir}")
    
    def save_results(self):
        """Save experimental results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"ablation_results_{timestamp}.yaml"
        
        with open(results_file, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        
        # Also save as CSV for easy analysis
        self._save_results_csv()
    
    def _save_results_csv(self):
        """Save results in CSV format for analysis"""
        data_rows = []
        
        for variant in self.variants:
            for problem_size in self.problem_sizes:
                if problem_size in self.results[variant]:
                    results = self.results[variant][problem_size]
                    
                    for i in range(len(results['solution_cost'])):
                        data_rows.append({
                            'variant': variant,
                            'problem_size': problem_size,
                            'instance': i,
                            'solution_cost': results['solution_cost'][i],
                            'computation_time': results['computation_time'][i],
                            'memory_usage': results['memory_usage'][i]
                        })
        
        df = pd.DataFrame(data_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.results_dir / f"ablation_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        return df
    
    def analyze_results(self):
        """Analyze experimental results"""
        print("\\n=== Analysis ===")
        
        # Load results into DataFrame
        df = self._save_results_csv()
        
        # Summary statistics
        summary = df.groupby(['variant', 'problem_size'])['solution_cost'].agg(['mean', 'std']).reset_index()
        print("\\nSummary Statistics (Solution Cost):")
        print(summary)
        
        # Percentage improvements
        print("\\nPerformance Improvements:")
        baseline_results = df[df['variant'] == '0_baseline'].groupby('problem_size')['solution_cost'].mean()
        
        for variant in ['1_static_rl', '2_dynamic_rl']:
            variant_results = df[df['variant'] == variant].groupby('problem_size')['solution_cost'].mean()
            improvements = (baseline_results - variant_results) / baseline_results * 100
            
            print(f"\\n{variant} vs baseline:")
            for size in self.problem_sizes:
                if size in improvements.index:
                    print(f"  {size} nodes: {improvements[size]:.1f}% improvement")
        
        # Statistical significance tests
        self._statistical_tests(df)
    
    def _statistical_tests(self, df: pd.DataFrame):
        """Perform statistical significance tests"""
        from scipy import stats
        
        print("\\n=== Statistical Tests ===")
        
        for problem_size in self.problem_sizes:
            print(f"\\nProblem size: {problem_size}")
            
            # Get data for this problem size
            size_data = df[df['problem_size'] == problem_size]
            
            # Pairwise comparisons
            variants = ['0_baseline', '1_static_rl', '2_dynamic_rl']
            for i in range(len(variants)):
                for j in range(i+1, len(variants)):
                    variant1, variant2 = variants[i], variants[j]
                    
                    data1 = size_data[size_data['variant'] == variant1]['solution_cost'].values
                    data2 = size_data[size_data['variant'] == variant2]['solution_cost'].values
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(data1, data2)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        
                        print(f"  {variant1} vs {variant2}: t={t_stat:.3f}, p={p_value:.4f} {significance}")
    
    def generate_plots(self):
        """Generate visualization plots"""
        print("\\nGenerating plots...")
        
        # Load data
        df = pd.read_csv(list(self.results_dir.glob("ablation_results_*.csv"))[-1])
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        fig_dir = self.results_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison bar plot
        self._plot_performance_comparison(df, fig_dir)
        
        # 2. Scalability analysis
        self._plot_scalability_analysis(df, fig_dir)
        
        # 3. Computation time comparison
        self._plot_computation_time(df, fig_dir)
        
        print(f"Plots saved to: {fig_dir}")
    
    def _plot_performance_comparison(self, df: pd.DataFrame, fig_dir: Path):
        """Plot performance comparison between variants"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ablation Study: Model Variant Comparison', fontsize=16)
        
        variant_names = {
            '0_baseline': 'Greedy Baseline',
            '1_static_rl': 'Static RL',
            '2_dynamic_rl': 'Dynamic RL'
        }
        
        for idx, problem_size in enumerate(self.problem_sizes[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            size_data = df[df['problem_size'] == problem_size]
            
            # Box plot
            size_data['variant_name'] = size_data['variant'].map(variant_names)
            sns.boxplot(data=size_data, x='variant_name', y='solution_cost', ax=ax)
            ax.set_title(f'{problem_size} Nodes')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, df: pd.DataFrame, fig_dir: Path):
        """Plot scalability analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Solution cost vs problem size
        summary = df.groupby(['variant', 'problem_size'])['solution_cost'].agg(['mean', 'std']).reset_index()
        
        variant_names = {
            '0_baseline': 'Greedy Baseline',
            '1_static_rl': 'Static RL', 
            '2_dynamic_rl': 'Dynamic RL'
        }
        
        for variant in self.variants:
            variant_data = summary[summary['variant'] == variant]
            ax1.errorbar(variant_data['problem_size'], variant_data['mean'], 
                        yerr=variant_data['std'], marker='o', label=variant_names[variant])
        
        ax1.set_xlabel('Problem Size (nodes)')
        ax1.set_ylabel('Solution Cost')
        ax1.set_title('Scalability: Solution Cost vs Problem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Computation time vs problem size
        time_summary = df.groupby(['variant', 'problem_size'])['computation_time'].agg(['mean', 'std']).reset_index()
        
        for variant in self.variants:
            variant_data = time_summary[time_summary['variant'] == variant]
            ax2.errorbar(variant_data['problem_size'], variant_data['mean'],
                        yerr=variant_data['std'], marker='s', label=variant_names[variant])
        
        ax2.set_xlabel('Problem Size (nodes)')
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Scalability: Computation Time vs Problem Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_computation_time(self, df: pd.DataFrame, fig_dir: Path):
        """Plot computation time comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variant_names = {
            '0_baseline': 'Greedy Baseline',
            '1_static_rl': 'Static RL',
            '2_dynamic_rl': 'Dynamic RL'
        }
        
        df['variant_name'] = df['variant'].map(variant_names)
        
        sns.boxplot(data=df, x='problem_size', y='computation_time', hue='variant_name', ax=ax)
        ax.set_xlabel('Problem Size (nodes)')
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Computation Time Comparison')
        ax.legend(title='Model Variant')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'computation_time.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run ablation study for Dynamic Graph Transformer CVRP')
    parser.add_argument('--config', type=str, default='configs/ablation_study.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run ablation study
    runner = AblationStudyRunner(args.config)
    runner.run_experiment()


if __name__ == "__main__":
    main()
