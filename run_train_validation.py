#!/usr/bin/env python3
"""
Thin CLI orchestrator for CVRP training/validation using refactored modules.
"""

import os
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.pipelines.train import train_all_models, train_one_model, set_seeds, generate_cvrp_instance
from src.models.pointer import BaselinePointerNetwork
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork
from src.models.gat import GraphAttentionTransformer


def setup_logging(config=None):
    level = logging.INFO
    format_str = '%(message)s'
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', '%(message)s')
        level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)


def model_key(name: str) -> str:
    mapping = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'GAT+RL (legacy)': 'gat_rl_legacy',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('+', '_').replace('-', '_'))


def save_histories_csv(results, config, base_dir, logger):
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    num_epochs = int(config['training']['num_epochs'])
    val_freq = int(config['training']['validation_frequency'])
    for model_name, data in results.items():
        hist = data['history']
        train_losses = hist.get('train_losses', [])
        train_costs = hist.get('train_costs', [])
        val_costs_seq = hist.get('val_costs', [])
        total_epochs = num_epochs + 1
        # Map val_costs to epochs (frequency, plus final epoch)
        val_costs = [float('nan')] * total_epochs
        idx = 0
        for ep in range(total_epochs):
            should_val = (ep % max(1, val_freq) == 0) or (ep == num_epochs)
            if should_val and idx < len(val_costs_seq):
                val_costs[ep] = val_costs_seq[idx]
                idx += 1
        df = pd.DataFrame({
            'epoch': list(range(total_epochs)),
            'train_loss': (train_losses + [float('nan')] * max(0, total_epochs - len(train_losses)))[:total_epochs],
            'train_cost': (train_costs + [float('nan')] * max(0, total_epochs - len(train_costs)))[:total_epochs],
            'val_cost': val_costs,
        })
        out = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
        df.to_csv(out, index=False)
        logger.info(f'🧾 Saved history CSV: {out}')


def save_models_and_analysis(results, training_times, models, config, base_dir, logger):
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    for model_name, model in models.items():
        path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name]['history'],
            'training_time': training_times[model_name],
        }, path)
        logger.info(f'💾 Saved model checkpoint: {path}')

    torch.save({
        'results': {k: v['history'] for k, v in results.items()},
        'training_times': training_times,
        'config': config,
    }, os.path.join(analysis_dir, 'comparative_study_complete.pt'))
    logger.info(f'📦 Saved comparative analysis: {os.path.join(analysis_dir, "comparative_study_complete.pt")}')


def parse_args():
    p = argparse.ArgumentParser(description='Train CVRP models (thin orchestrator)')
    p.add_argument('--config', type=str, default='configs/small.yaml', help='Path to YAML configuration file')
    p.add_argument('--include-legacy', action='store_true', help='Include legacy GAT+RL (requires ../GAT_RL and torch-geometric)')
    p.add_argument('--force-retrain', action='store_true', help='Force retraining even if outputs already exist')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    # Seeds and CPU-only environment
    set_seeds(cfg.get('experiment', {}).get('random_seed', 42))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    base_dir = str(Path(cfg.get('working_dir_path', 'results')).as_posix())
    os.makedirs(base_dir, exist_ok=True)

    # Build models on demand
    input_dim = cfg['model']['input_dim']
    hidden_dim = cfg['model']['hidden_dim']
    num_heads = cfg['model']['num_heads']
    num_layers = cfg['model']['num_layers']
    dropout = cfg['model']['transformer_dropout']
    ff_mult = cfg['model']['feedforward_multiplier']
    edge_div = cfg['model']['edge_embedding_divisor']

    def build_model(name: str):
        if name == 'Pointer+RL':
            return BaselinePointerNetwork(input_dim, hidden_dim, cfg)
        if name == 'GT-Greedy':
            return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'GT+RL':
            return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'DGT+RL':
            return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'GAT+RL':
            return GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_div, cfg)
        raise ValueError(f'Unknown model name: {name}')

    logger.info('🚀 Training models...')

    # Prepare output dirs
    csv_dir = os.path.join(base_dir, 'csv')
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pytorch_dir, exist_ok=True)

    # Helper: decide skip
    def should_skip(name: str) -> bool:
        if args.force_retrain:
            return False
        ckpt = os.path.join(pytorch_dir, f'model_{model_key(name)}.pt')
        hist = os.path.join(csv_dir, f'history_{model_key(name)}.csv')
        return os.path.exists(ckpt) and os.path.exists(hist)

    # If skip, reconstruct minimal results from CSV and checkpoint
    def load_minimal_results(name: str):
        ckpt = os.path.join(pytorch_dir, f'model_{model_key(name)}.pt')
        hist_path = os.path.join(csv_dir, f'history_{model_key(name)}.csv')
        training_time = float('nan')
        try:
            sd = torch.load(ckpt, map_location='cpu')
            training_time = float(sd.get('training_time', float('nan')))
        except Exception:
            pass
        final_val = float('nan')
        try:
            df = pd.read_csv(hist_path)
            if 'val_cost' in df.columns:
                series = df['val_cost'].dropna().tolist()
                if series:
                    final_val = float(series[-1])
        except Exception:
            pass
        history = {'train_losses': [], 'train_costs': [], 'val_costs': ([] if final_val != final_val else [final_val]), 'final_val_cost': final_val}
        return history, training_time

    # Train with skip logic (per-model)
    results = {}
    training_times = {}
    models = {}
    for name in ['Pointer+RL', 'GT-Greedy', 'GT+RL', 'DGT+RL', 'GAT+RL']:
        if should_skip(name):
            logger.info(f'⏭️  Skipping {name}: outputs already exist')
            hist, ttime = load_minimal_results(name)
            results[name] = {'history': hist}
            training_times[name] = ttime
            # Load model to populate parameter count
            try:
                m = build_model(name)
                state = torch.load(os.path.join(pytorch_dir, f'model_{model_key(name)}.pt'), map_location='cpu')
                sd = state.get('model_state_dict', state)
                m.load_state_dict(sd)
                models[name] = m
            except Exception:
                models[name] = build_model(name)
            continue
        # Train only this model
        m = build_model(name)
        hist, ttime, _ = train_one_model(m, name, cfg, logger.info)
        results[name] = {'history': hist}
        training_times[name] = ttime
        models[name] = m

    # Optionally include legacy model
    if args.include_legacy:
        try:
            # Check dependencies
            try:
                import torch_geometric  # noqa: F401
                from torch_geometric.loader import DataLoader
            except Exception as e:
                logger.warning(f"Legacy GAT+RL skipped: torch-geometric not available ({e})")
                raise
            try:
                from src_batch.model.Model import Model as LegacyGATModel
                from src_batch.train.train_model import train as legacy_train
            except Exception as e:
                logger.warning(f"Legacy GAT+RL skipped: src_batch/../GAT_RL not available ({e})")
                raise

            # Build legacy model
            lg_cfg = cfg.get('model', {}).get('legacy_gat', {})
            legacy_model = LegacyGATModel(
                node_input_dim=3,
                edge_input_dim=1,
                hidden_dim=lg_cfg.get('hidden_dim', cfg['model']['hidden_dim']),
                edge_dim=lg_cfg.get('edge_dim', 16),
                layers=lg_cfg.get('layers', cfg['model']['num_layers']),
                negative_slope=lg_cfg.get('negative_slope', 0.2),
                dropout=lg_cfg.get('dropout', 0.6),
            )

            # Generate dataset (match small config size)
            n_instances = int(cfg['training']['num_instances'])
            n_customers = int(cfg['problem']['num_customers'])
            capacity = int(cfg['problem']['vehicle_capacity'])
            coord_range = int(cfg['problem']['coord_range'])
            demand_range = list(cfg['problem']['demand_range'])

            # Build PyG Data list
            import torch as _torch
            from torch_geometric.data import Data
            def _to_pyg(inst):
                coords = _torch.tensor(inst['coords'], dtype=_torch.float32)
                n = coords.size(0)
                ii, jj = _torch.meshgrid(_torch.arange(n), _torch.arange(n), indexing='ij')
                edge_index = _torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0)
                edge_attr = _torch.tensor(inst['distances'].reshape(-1, 1), dtype=_torch.float32)
                demand = _torch.tensor(inst['demands'], dtype=_torch.float32).unsqueeze(1)
                capacity_t = _torch.tensor([inst['capacity']], dtype=_torch.float32)
                return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr, demand=demand, capacity=capacity_t)

            data_list = [_to_pyg(generate_cvrp_instance(n_customers, capacity, coord_range, demand_range, seed=i)) for i in range(n_instances)]
            split = int(float(cfg['training']['train_val_split']) * n_instances)
            train_loader = DataLoader(data_list[:split], batch_size=int(cfg['training']['batch_size']), shuffle=False)
            val_loader = DataLoader(data_list[split:], batch_size=int(cfg['training']['batch_size']), shuffle=False)

            # Legacy training params
            tadv = cfg.get('training_advanced', {}).get('legacy_gat', {})
            lr = float(tadv.get('learning_rate', cfg['training']['learning_rate']))
            T = float(tadv.get('temperature', cfg['inference']['default_temperature']))
            n_steps = int(n_customers * int(tadv.get('max_steps_multiplier', cfg['inference']['max_steps_multiplier'])))
            num_epochs = int(cfg['training']['num_epochs'])

            # Train legacy
            logger.info('🚀 Training legacy GAT+RL...')
            ckpt_folder = os.path.join(base_dir, 'checkpoints', 'legacy')
            os.makedirs(ckpt_folder, exist_ok=True)
            t0 = time.time()
            legacy_train(legacy_model, train_loader, val_loader, ckpt_folder, 'actor.pt', lr, n_steps, num_epochs, T)
            legacy_time = time.time() - t0

            # Evaluate final val cost per customer (greedy)
            legacy_model.eval()
            val_costs = []
            with torch.no_grad():
                for batch in val_loader:
                    actions, _logp = legacy_model(batch, n_steps=n_steps, greedy=True, T=T)
                    depot = _torch.zeros(actions.size(0), 1, dtype=_torch.long)
                    full = _torch.cat([depot, actions, depot], dim=1)
                    # Per-graph coords slicing
                    ptr = batch.ptr if hasattr(batch, 'ptr') else None
                    for b in range(full.size(0)):
                        route = full[b].cpu().tolist()
                        if ptr is not None:
                            start = int(ptr[b].item()); end = int(ptr[b+1].item())
                            coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                        else:
                            coords = batch.x.view(-1, 2).cpu().numpy()
                        # Euclidean cost
                        csum = 0.0
                        for i in range(len(route) - 1):
                            a = route[i]; c = route[i+1]
                            pa = coords[a]; pb = coords[c]
                            csum += float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
                        val_costs.append(csum / float(n_customers))
            final_val = float(np.mean(val_costs)) if val_costs else float('nan')

            # Write legacy CSV history (sparse val only at final epoch)
            csv_dir = os.path.join(base_dir, 'csv')
            os.makedirs(csv_dir, exist_ok=True)
            total_epochs = num_epochs + 1
            import pandas as _pd
            df_hist = _pd.DataFrame({
                'epoch': list(range(total_epochs)),
                'train_loss': [float('nan')]*total_epochs,
                'train_cost': [float('nan')]*total_epochs,
                'val_cost': [float('nan')]*total_epochs,
            })
            df_hist.loc[num_epochs, 'val_cost'] = final_val
            out_hist = os.path.join(csv_dir, f'history_{model_key("GAT+RL (legacy)")}.csv')
            df_hist.to_csv(out_hist, index=False)
            logger.info(f'🧾 Saved legacy history CSV: {out_hist}')

            # Save legacy checkpoint
            legacy_ckpt = os.path.join(base_dir, 'pytorch', 'model_gat_rl_legacy.pt')
            os.makedirs(os.path.dirname(legacy_ckpt), exist_ok=True)
            torch.save({'model_state_dict': legacy_model.state_dict(), 'model_name': 'GAT+RL (legacy)', 'config': cfg, 'results': {'final_val_cost': final_val}, 'training_time': legacy_time}, legacy_ckpt)
            logger.info(f'💾 Saved legacy model checkpoint: {legacy_ckpt}')

            # Add to results for analysis
            results['GAT+RL (legacy)'] = {'history': {'train_losses': [], 'train_costs': [], 'val_costs': [final_val], 'final_val_cost': final_val}}
            training_times['GAT+RL (legacy)'] = legacy_time
            models['GAT+RL (legacy)'] = legacy_model
        except Exception:
            # Dependencies missing or repo unavailable; already warned
            pass

    # Save artifacts after (possibly) adding legacy
    save_histories_csv(results, cfg, base_dir, logger)
    save_models_and_analysis(results, training_times, models, cfg, base_dir, logger)

    # Summary
    logger.info('\n📊 SUMMARY')
    for name, data in results.items():
        params = sum(p.numel() for p in models[name].parameters())
        final_val = data['history']['final_val_cost']
        logger.info(f'- {name}: params={params:,}, time={training_times[name]:.1f}s, final_val={final_val:.3f}/cust')


if __name__ == '__main__':
    main()

