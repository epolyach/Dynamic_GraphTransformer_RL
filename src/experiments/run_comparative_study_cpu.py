#!/usr/bin/env python3
"""
CPU Comparative Study Orchestrator (training + evaluation + plots)

Goals
- Intuitive UX like Ver1: one command to train missing models (on CPU),
  reuse cached results if available, and produce comparison tables/plots.
- Support models:
  * dynamic_gt_rl (train via unified pipeline)
  * static_rl (train via unified pipeline)
  * pointer_rl (train via unified pipeline; optional/legacy)
  * greedy_baseline (eval-only heuristic)
  * naive_baseline (eval-only roundtrip)
  * (stubs reserved for gat_rl and gat_rl_legacy)

Behavior
- For RL models, the script looks for a cached summary JSON under the unified
  trainer's output directory. If not found (or if forced), it triggers training via
  src/training/pipeline_train.train_pipeline, then writes the summary JSON.
- For baselines, it evaluates on the same deterministic validation set used
  by the unified trainer (derived from --seed and --instances).
- Writes a consolidated CSV and a plot to match Ver1-style UX.

Outputs
- CSV: results/comparative_study_cpu.csv
- Plot: utils/plots/comparative_study_results.png
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

# Project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    sys.path.append(str(project_root / 'src'))

from src.models import (
    GreedyGraphTransformerBaseline,
)
from src.training.pipeline_train import train_pipeline
from src.utils.RL.euclidean_cost import euclidean_cost


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_instance(num_nodes: int, capacity: float, device: torch.device, seed: int = None) -> Data:
    # Coordinates in unit square on a discrete 1..100 grid, divided by 100
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        coords = (torch.randint(1, 101, (num_nodes, 2), generator=g, device=device, dtype=torch.int64).float() / 100.0)
        demands = (torch.randint(1, 11, (num_nodes, 1), generator=g, device=device, dtype=torch.int64).float())
    else:
        coords = (torch.randint(1, 101, (num_nodes, 2), device=device, dtype=torch.int64).float() / 100.0)
        demands = torch.randint(1, 11, (num_nodes, 1), device=device, dtype=torch.int64).float()
    # Customer demands sampled from {0.1, 0.2, ..., 1.0}; depot demand = 0
    demands[0] = 0.0

    # Fully connected directed edge attributes as Euclidean distances
    edge_idx = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge_idx.append([i, j])
            d = torch.norm(coords[i] - coords[j], dim=0, keepdim=True)
            edge_attr.append(d)
    edge_index = torch.tensor(edge_idx, device=device).t().long() if edge_idx else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 1), device=device)

    data = Data(
        x=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand=demands,
        capacity=torch.full((num_nodes,), capacity, device=device),
        batch=torch.zeros(num_nodes, dtype=torch.long, device=device),
    )
    data.num_graphs = 1
    return data


def make_batch(instances: List[Data]) -> Data:
    device = instances[0].x.device
    x_list, edge_index_list, edge_attr_list, demand_list, capacity_list, batch_vec = [], [], [], [], [], []
    offset = 0
    for b, d in enumerate(instances):
        n = d.x.size(0)
        x_list.append(d.x)
        demand_list.append(d.demand)
        capacity_list.append(d.capacity)
        batch_vec.append(torch.full((n,), b, dtype=torch.long, device=device))
        if d.edge_index.numel() > 0:
            edge_index_list.append(d.edge_index + offset)
            edge_attr_list.append(d.edge_attr)
        offset += n
    x = torch.cat(x_list, dim=0)
    demand = torch.cat(demand_list, dim=0)
    capacity = torch.cat(capacity_list, dim=0)
    batch = torch.cat(batch_vec, dim=0)
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), device=device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, demand=demand, capacity=capacity, batch=batch)
    data.num_graphs = len(instances)
    return data


def naive_roundtrip_cost(coords_np: np.ndarray) -> float:
    depot = coords_np[0]
    dists = np.linalg.norm(coords_np[1:] - depot, axis=1)
    return float(2.0 * dists.sum())


def eval_greedy_baseline_on_val(customers: int, capacity: float, seed: int, instances: int, batch_size: int, device: torch.device) -> Tuple[float, float]:
    """Evaluate the GreedyGraphTransformerBaseline on the same validation seeds
    as the unified trainer (first val_count seeds starting from data_seed).
    Returns (avg_cost_per_customer, avg_time_seconds).
    """
    # Match pipeline's validation split logic
    val_count = max(32, min(128, instances // 5))
    seeds = [seed + i for i in range(val_count)]

    model = GreedyGraphTransformerBaseline().to(device)
    model.eval()
    n_nodes = customers + 1
    n_steps = customers + 6

    costs = []
    times = []
    with torch.no_grad():
        total_batches = (len(seeds) + batch_size - 1) // batch_size
        for bi, i in enumerate(range(0, len(seeds), batch_size), start=1):
            batch_seeds = seeds[i:i + batch_size]
            inst_list = [make_instance(n_nodes, capacity, device, s) for s in batch_seeds]
            batch_data = make_batch(inst_list)
            print(f"[eval] greedy_baseline batch {bi}/{total_batches} (val instances {i+1}-{i+len(batch_seeds)})")
            t0 = time.perf_counter()
            actions, _ = model(batch_data, n_steps=n_steps, greedy=True)
            t1 = time.perf_counter()
            c = euclidean_cost(batch_data.x, actions, batch_data)
            costs.append(c.detach().cpu())
            times.append(t1 - t0)
    cost_tensor = torch.cat(costs) if costs else torch.tensor([], dtype=torch.float32)
    avg_cost_per_customer = float(cost_tensor.mean().item()) / max(1, customers)
    avg_time = float(np.mean(times)) if times else float('nan')
    return avg_cost_per_customer, avg_time


def ensure_trained_and_load_summary(model: str, customers: int, instances: int, epochs: int, batch: int, capacity: float, lr: float, seed: int, base_out: Path, force: bool = False) -> Dict[str, float]:
    """Train via unified pipeline if summary cache is missing (or forced). Return summary dict."""
    device = torch.device('cpu')
    out_dir = base_out / f"cpu_{model}_C{customers}_I{instances}_E{epochs}_B{batch}"
    summary_path = out_dir / 'summary.json'
    if summary_path.exists() and not force:
        with summary_path.open('r') as f:
            print(f"[orchestrator] Reusing cached run: {summary_path}")
            return json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Train
    print(f"[orchestrator] Training {model} on CPU: C={customers} I={instances} E={epochs} B={batch} lr={lr} seed={seed}")
    t0 = time.perf_counter()
    summary = train_pipeline(
        model_name=model,
        device=device,
        use_amp=False,
        customers=customers,
        instances=instances,
        epochs=epochs,
        batch_size=batch,
        capacity=capacity,
        lr=lr,
        data_seed=seed,
        out_dir=out_dir,
    )
    t1 = time.perf_counter()
    print(f"[orchestrator] Finished training {model} in {t1 - t0:.2f}s; best val/cust={summary['best_val_cost_per_customer']:.4f}")
    enriched = {
        'model': model,
        'customers': customers,
        'instances': instances,
        'epochs': epochs,
        'batch': batch,
        'capacity': capacity,
        'lr': lr,
        'data_seed': seed,
        'best_val_cost_per_customer': summary['best_val_cost_per_customer'],
        'train_time_seconds': t1 - t0,
        'out_dir': str(out_dir),
    }
    with summary_path.open('w') as f:
        json.dump(enriched, f, indent=2)
    return enriched


def generate_plots(df: pd.DataFrame, out_plot: Path):
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    # Bar plot of Val/Cust for all models present
    order = ['dynamic_gt_rl', 'static_rl', 'pointer_rl', 'greedy_baseline', 'naive_baseline', 'gat_rl', 'gat_rl_legacy']
    df_plot = df.copy()
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=order, ordered=True)
    df_plot = df_plot.sort_values('Model')

    plt.figure(figsize=(10, 6))
    plt.bar(df_plot['Model'], df_plot['Val/Cust'])
    plt.ylabel('Average Cost per Customer')
    plt.title('CPU Comparative Study')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved plot to {out_plot}")


def parse_routes_into_tours(route: List[int], demands: np.ndarray) -> List[Dict]:
    """Parse a route into individual tours, identifying roundtrips vs multi-customer tours."""
    tours = []
    current_tour = []
    tour_demand = 0.0
    
    for i, node in enumerate(route):
        if node == 0:  # depot
            if current_tour:  # end of a tour
                # Check if this was a roundtrip (only one customer)
                is_roundtrip = len(current_tour) == 1
                tours.append({
                    'nodes': [0] + current_tour + [0],
                    'demand': tour_demand,
                    'is_roundtrip': is_roundtrip
                })
                current_tour = []
                tour_demand = 0.0
        else:  # customer
            if node not in current_tour:  # avoid duplicates
                current_tour.append(node)
                tour_demand += demands[node]
    
    # Handle case where route doesn't end at depot
    if current_tour:
        is_roundtrip = len(current_tour) == 1
        tours.append({
            'nodes': [0] + current_tour + [0],
            'demand': tour_demand,
            'is_roundtrip': is_roundtrip
        })
    
    return tours



def validate_cvrp_route(route: List[int], demands: np.ndarray, capacity: float, n_customers: int) -> Dict:
    """Comprehensive CVRP route validation function."""
    result = {'valid': True, 'errors': [], 'warnings': [], 'stats': {}}
    
    if not route or route[0] != 0 or route[-1] != 0:
        result['valid'] = False
        result['errors'].append("Route must start and end at depot")
        return result
    
    # Check all customers served exactly once
    customers_in_route = [node for node in route if node != 0]
    unique_customers = set(customers_in_route)
    expected_customers = set(range(1, n_customers + 1))
    
    if len(customers_in_route) != len(unique_customers):
        result['valid'] = False
        result['errors'].append("Duplicate customers served")
    
    if unique_customers != expected_customers:
        missing = expected_customers - unique_customers
        extra = unique_customers - expected_customers
        if missing:
            result['valid'] = False
            result['errors'].append(f"Missing customers: {missing}")
        if extra:
            result['valid'] = False
            result['errors'].append(f"Invalid customers: {extra}")
    
    # Capacity validation - split route into tours
    tours = []
    current_tour = []
    
    for node in route:
        if node == 0 and current_tour:
            tours.append(current_tour)
            current_tour = []
        elif node != 0:
            current_tour.append(node)
    
    if current_tour:
        tours.append(current_tour)
    
    capacity_violations = []
    for tour_idx, tour in enumerate(tours):
        tour_demand = int(sum(demands[customer] for customer in tour))  # Integer sum
        if tour_demand > int(capacity):  # Exact integer comparison
            capacity_violations.append({
                'tour': tour_idx + 1,
                'nodes': tour,
                'demand': tour_demand,
                'violation': tour_demand - capacity
            })
    
    if capacity_violations:
        result['valid'] = False
        result['errors'].append(f"Capacity violations in {len(capacity_violations)} tours")
        result['capacity_violations'] = capacity_violations
    
    result['stats'] = {
        'num_tours': len(tours),
        'depot_visits': route.count(0),
        'tour_demands': [sum(demands[c] for c in tour) for tour in tours]
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='CPU Comparative Study Orchestrator (train missing, reuse cached, make plots)')
    parser.add_argument('--models', nargs='+', default=['dynamic_gt_rl', 'static_rl', 'greedy_baseline', 'naive_baseline'],
                        help='Models to include')
    parser.add_argument('--customers', type=int, default=20, help='Number of customers (excluding depot)')
    parser.add_argument('--instances', type=int, default=800, help='Instances for training (unified pipeline)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--capacity', type=float, default=30.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=12345, help='Base seed for deterministic instances')
    parser.add_argument('--recalculate_rl_weights', action='store_true', help='Force retrain RL models (ignore cache)')
    parser.add_argument('--train_out_dir', type=str, default='results_train', help='Where unified trainer stores runs')
    parser.add_argument('--out_csv', type=str, default='results/comparative_study_cpu.csv')
    parser.add_argument('--out_plot', type=str, default='utils/plots/comparative_study_results.png')
    parser.add_argument('--test_seed', type=int, default=20250809, help='Seed for single test instance application step')
    args = parser.parse_args()

    device = torch.device('cpu')
    print("[orchestrator] Starting comparative study")
    print(f"[orchestrator] Models: {' '.join(args.models)}")
    print(f"[orchestrator] Params: customers={args.customers} instances={args.instances} epochs={args.epochs} batch={args.batch} lr={args.lr} seed={args.seed}")

    # Collect rows
    rows: List[Dict[str, object]] = []
    base_out = project_root / args.train_out_dir

    # Baselines depend on validation set; evaluate once if requested
    if 'naive_baseline' in args.models or 'greedy_baseline' in args.models:
        val_count = max(32, min(128, args.instances // 5))
        seeds = [args.seed + i for i in range(val_count)]
        n_nodes = args.customers + 1
        # Prepare val instances
        val_graphs = [make_instance(n_nodes, args.capacity, device, s) for s in seeds]
        # Naive baseline
        if 'naive_baseline' in args.models:
            print("[orchestrator] Evaluating naive_baseline on deterministic validation set")
            naive_vals = []
            for g in val_graphs:
                coords = g.x.detach().cpu().numpy()
                naive_vals.append(naive_roundtrip_cost(coords) / max(1, args.customers))
            rows.append({
                'Model': 'naive_baseline',
                'Val/Cust': float(np.mean(naive_vals)),
                'CPU Time (s)': None,
                'OutDir': None,
            })
        # Greedy baseline
        if 'greedy_baseline' in args.models:
            print("[orchestrator] Evaluating greedy_baseline on deterministic validation set")
            avg_cost, avg_time = eval_greedy_baseline_on_val(args.customers, args.capacity, args.seed, args.instances, args.batch, device)
            rows.append({
                'Model': 'greedy_baseline',
                'Val/Cust': float(avg_cost),
                'CPU Time (s)': float(avg_time),
                'OutDir': None,
            })

    # RL models via unified trainer
    for model in ['dynamic_gt_rl', 'static_rl', 'pointer_rl']:
        if model not in args.models:
            continue
        summary = ensure_trained_and_load_summary(
            model=model,
            customers=args.customers,
            instances=args.instances,
            epochs=args.epochs,
            batch=args.batch,
            capacity=args.capacity,
            lr=args.lr,
            seed=args.seed,
            base_out=base_out,
            force=args.recalculate_rl_weights,
        )
        rows.append({
            'Model': model,
            'Val/Cust': float(summary['best_val_cost_per_customer']),
            'CPU Time (s)': float(summary['train_time_seconds']),
            'OutDir': summary['out_dir'],
        })

    # TODO: Optional integration for 'gat_rl' and 'gat_rl_legacy' via ../GAT_RL
    # Placeholder: skip unless explicitly implemented.

    df = pd.DataFrame(rows)
    out_csv = project_root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved results table to {out_csv}")

    out_plot = project_root / args.out_plot
    generate_plots(df, out_plot)

    # Apply models to test instance and generate final annotated plots
    try:
        print("[orchestrator] Applying trained models to test instance and generating annotated plots")
        # Deterministic test instance
        customers = args.customers
        n = customers + 1
        device = torch.device('cpu')
        g = torch.Generator(device=device); g.manual_seed(args.test_seed)
        coords = (torch.randint(1, 101, (n, 2), generator=g, device=device, dtype=torch.int64).float() / 100.0)
        demands = (torch.randint(1, 11, (n, 1), generator=g, device=device, dtype=torch.int64).float())
        demands[0] = 0.0
        edge_idx = []; edge_attr = []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                edge_idx.append([i, j])
                edge_attr.append(torch.norm(coords[i] - coords[j]).unsqueeze(0))
        edge_index = torch.tensor(edge_idx, device=device).t().long()
        edge_attr = torch.stack(edge_attr)
        inst = Data(x=coords, edge_index=edge_index, edge_attr=edge_attr,
                    demand=demands, capacity=torch.full((n,), args.capacity, device=device),
                    batch=torch.zeros(n, dtype=torch.long, device=device))
        inst.num_graphs = 1

        def to_idx_seq(actions):
            act = actions.squeeze()
            if act.dim() > 1:
                act = act[:, 0]
            return [int(i) for i in act.tolist()]

        def build_feasible_route(idx_seq, demands_tensor, capacity_val):
            d = demands_tensor.view(-1).tolist()
            N = len(d)
            served = set()
            route = [0]
            rem = float(capacity_val)
            
            # Process the sequence more carefully
            for idx in idx_seq:
                if idx == 0:
                    if route[-1] != 0:
                        route.append(0)
                        rem = float(capacity_val)
                    continue
                if idx <= 0 or idx >= N or idx in served:
                    continue
                dem = float(d[idx])
                # Exact integer capacity check
                if int(dem) <= int(rem):
                    route.append(idx)
                    served.add(idx)
                    rem -= dem
                else:
                    # Must return to depot first
                    if route[-1] != 0:
                        route.append(0)
                    rem = float(capacity_val)
                    # Check again with fresh capacity
                    if int(dem) <= int(rem):
                        route.append(idx)
                        served.add(idx)
                        rem -= dem
            
            # Serve any remaining customers with strict capacity checking
            remaining = [i for i in range(1, N) if i not in served]
            for idx in remaining:
                dem = float(d[idx])
                # Always return to depot for remaining customers to ensure capacity
                if route[-1] != 0:
                    route.append(0)
                rem = float(capacity_val)
                if int(dem) <= int(rem):
                    route.append(idx)
                    served.add(idx)
                    rem -= dem
            
            if route[-1] != 0:
                route.append(0)
            
            # Remove consecutive depots
            comp = [route[0]]
            for x in route[1:]:
                if not (x == 0 and comp[-1] == 0):
                    comp.append(x)
            return comp

        plots_dir = project_root / 'utils' / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        c = coords.detach().cpu().numpy()
        dvals = demands.detach().cpu().numpy().reshape(-1)
        base, scale = 20.0, 30.0
        sizes = base + scale * dvals

        # Import models and build solutions
        from src.models import StaticRLGraphTransformer, DynamicGraphTransformerModel
        spec = {
            'dynamic_gt_rl': DynamicGraphTransformerModel,
            'static_rl': StaticRLGraphTransformer,
            'greedy_baseline': GreedyGraphTransformerBaseline,
        }
        
        # Color palette for different tours
        tour_colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for name in args.models:
            if name == 'naive_baseline' or name not in spec:
                continue
            model = spec[name]()
            # Load checkpoint if available
            ckpt_path = project_root / args.train_out_dir / f"cpu_{name}_C{args.customers}_I{args.instances}_E{args.epochs}_B{args.batch}" / 'checkpoint.pt'
            if ckpt_path.exists():
                try:
                    state = torch.load(ckpt_path, map_location='cpu')
                    state = state.get('model_state_dict', state)
                    model.load_state_dict(state, strict=False)
                    print(f"[apply] Loaded checkpoint for {name}: {ckpt_path}")
                except Exception as e:
                    print(f"[apply] WARNING: failed to load checkpoint for {name}: {e}")
            model.eval()
            with torch.no_grad():
                actions, _ = model(inst, n_steps=customers + 6, greedy=True)
            route = build_feasible_route(to_idx_seq(actions), demands, args.capacity)
            
            # Validate the route
            validation_result = validate_cvrp_route(route, dvals, args.capacity, customers)
            if not validation_result['valid']:
                print(f"[apply] WARNING: {name} produced invalid route: {validation_result['errors']}")
                if 'capacity_violations' in validation_result:
                    for violation in validation_result['capacity_violations']:
                        print(f"  Tour {violation['tour']}: demand {violation['demand']:.2f} > capacity {args.capacity}")
            else:
                print(f"[apply] {name} route is valid ({validation_result['stats']['num_tours']} tours)")
            
            # Calculate cost directly from the displayed route
            route_cost = 0.0
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                route_cost += float(torch.norm(coords[a] - coords[b]).item())
            cost_pc = route_cost / customers
            
            # Parse route into tours
            tours = parse_routes_into_tours(route, dvals)
            
            # Create annotated plot
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Plot tours with different colors and annotations
            for tour_idx, tour in enumerate(tours):
                tour_nodes = tour['nodes']
                tour_demand = tour['demand']
                is_roundtrip = tour['is_roundtrip']
                
                # Choose color: grey for roundtrips, cycling colors for multi-customer tours
                if is_roundtrip:
                    color = 'grey'
                    alpha = 0.6
                    linewidth = 1.2
                else:
                    color = tour_colors[tour_idx % len(tour_colors)]
                    alpha = 0.85
                    linewidth = 1.6
                
                # Plot tour lines
                for i in range(len(tour_nodes) - 1):
                    a, b = tour_nodes[i], tour_nodes[i + 1]
                    ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], '-', 
                           color=color, alpha=alpha, linewidth=linewidth)
                
                # Annotate multi-customer tours with demand
                if not is_roundtrip and len(tour_nodes) > 3:  # More than depot-customer-depot
                    # Find midpoint of tour for annotation
                    customer_nodes = [node for node in tour_nodes if node != 0]
                    if customer_nodes:
                        mid_x = np.mean([c[node, 0] for node in customer_nodes])
                        mid_y = np.mean([c[node, 1] for node in customer_nodes])
                        ax.annotate(f'{int(tour_demand)}', (mid_x, mid_y), 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                                   fontsize=8, ha='center', va='center', color='white', fontweight='bold')
            
            # Plot depot
            ax.scatter(c[0, 0], c[0, 1], marker='*', s=180, c='red', edgecolors='black', linewidths=1.0)
            ax.annotate('(0,0.0)', (c[0, 0], c[0, 1]), textcoords='offset points', xytext=(6, 6), fontsize=9, color='black')
            
            # Plot customers
            ax.scatter(c[1:, 0], c[1:, 1], s=sizes[1:], c='lightblue', edgecolors='black', linewidths=0.9)
            for i in range(1, n):
                ax.annotate(f'({i},{int(dvals[i])})', (c[i, 0], c[i, 1]), textcoords='offset points', xytext=(6, 6), fontsize=8, color='black')
            
            ax.set_title(f'{name}  (cost/customer={cost_pc:.3f})')
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            out_png = plots_dir / f'test_instance_route_{name}.png'
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            
            # Save route-only JSON
            (plots_dir / f'test_instance_route_{name}.json').write_text(json.dumps(route))
            print(f"[apply] Generated {out_png}")
    except Exception as e:
        print(f"[apply] WARNING: failed to generate annotated plots: {e}")


if __name__ == '__main__':
    main()
