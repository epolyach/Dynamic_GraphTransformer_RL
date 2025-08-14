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


def write_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Write one model's history to CSV immediately (single source of truth).
    Includes epoch 0 (index 0 of histories) and a final row at epoch=num_epochs.
    """
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    num_epochs = int(config['training']['num_epochs'])
    val_freq = int(config['training']['validation_frequency'])
    train_losses = list(history.get('train_losses', []))
    train_costs = list(history.get('train_costs', []))
    val_costs_seq = list(history.get('val_costs', []))
    # We will produce rows for epochs 0..num_epochs (inclusive)
    total_rows = num_epochs + 1
    train_loss_series = [float('nan')] * total_rows
    train_cost_series = [float('nan')] * total_rows
    val_cost_series = [float('nan')] * total_rows
    # Map training series directly: index 0 -> epoch 0, ..., index num_epochs-1 -> epoch num_epochs-1
    for i, v in enumerate(train_losses):
        if i < num_epochs:
            train_loss_series[i] = v
    for i, v in enumerate(train_costs):
        if i < num_epochs:
            train_cost_series[i] = v
    # Carry forward last known values to fill epoch=num_epochs for display symmetry
    if num_epochs > 0 and (train_loss_series[num_epochs] != train_loss_series[num_epochs]):
        train_loss_series[num_epochs] = train_loss_series[num_epochs - 1]
    if num_epochs > 0 and (train_cost_series[num_epochs] != train_cost_series[num_epochs]):
        train_cost_series[num_epochs] = train_cost_series[num_epochs - 1]
    # Map validation costs by aligning to actual validation epochs (includes epoch 0 if present)
    expected_val_epochs = [ep for ep in range(0, num_epochs + 1)
                           if (ep % max(1, val_freq) == 0) or (ep == num_epochs)]
    for i, ep in enumerate(expected_val_epochs):
        if i < len(val_costs_seq):
            val_cost_series[ep] = val_costs_seq[i]
    # Ensure final epoch gets the last available validation cost explicitly
    if len(val_costs_seq) > 0:
        val_cost_series[num_epochs] = val_costs_seq[-1]
    df = pd.DataFrame({
        'epoch': list(range(0, num_epochs + 1)),
        'train_loss': train_loss_series,
        'train_cost': train_cost_series,
        'val_cost': val_cost_series,
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
        # Disable skip: always retrain to avoid partial/minimal histories or placeholders
        return False

    # If skip, reconstruct minimal results from CSV and checkpoint
    def load_minimal_results(name: str):
        ckpt = os.path.join(pytorch_dir, f'model_{model_key(name)}.pt')
        hist_path = os.path.join(csv_dir, f'history_{model_key(name)}.csv')
        analysis_path = os.path.join(base_dir, 'analysis', 'comparative_study_complete.pt')
        training_time = float('nan')
        # Try load training time from checkpoint
        try:
            sd = torch.load(ckpt, map_location='cpu')
            training_time = float(sd.get('training_time', float('nan')))
        except Exception:
            pass
        # Prefer full history from analysis file if available
        try:
            blob = torch.load(analysis_path, map_location='cpu')
            res = blob.get('results', {})
            if name in res and isinstance(res[name], dict):
                hist = res[name]
                # Ensure keys exist
                hist.setdefault('train_losses', [])
                hist.setdefault('train_costs', [])
                hist.setdefault('val_costs', [])
                hist.setdefault('final_val_cost', float('nan'))
                return hist, training_time
        except Exception:
            pass
        # Fallback to minimal history from CSV final value
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
            # Write CSV immediately from loaded history if present
            try:
                write_history_csv(name, results[name]['history'], cfg, base_dir, logger)
            except Exception:
                pass
            continue
        # Train only this model
        m = build_model(name)
        hist, ttime, _ = train_one_model(m, name, cfg, logger.info)
        results[name] = {'history': hist}
        training_times[name] = ttime
        models[name] = m
        # Write CSV immediately for this model
        write_history_csv(name, hist, cfg, base_dir, logger)

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
            # Capture stdout during legacy training to parse per-epoch mean loss/reward
            import io, re, sys
            buf = io.StringIO()
            # Tee stdout so we can both display logs live and capture them for parsing
            class _Tee(io.TextIOBase):
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, s):
                    for st in self.streams:
                        try:
                            st.write(s)
                        except Exception:
                            pass
                    return len(s)
                def flush(self):
                    for st in self.streams:
                        try:
                            st.flush()
                        except Exception:
                            pass
            tee = _Tee(sys.stdout, buf)
            old_stdout = sys.stdout
            t0 = time.time()
            try:
                sys.stdout = tee
                legacy_train(legacy_model, train_loader, val_loader, ckpt_folder, 'actor.pt', lr, n_steps, num_epochs, T)
            finally:
                sys.stdout = old_stdout
            legacy_time = time.time() - t0
            legacy_stdout = buf.getvalue()

            # Build per-epoch series for legacy by evaluating saved checkpoints
            import glob as _glob
            total_epochs = num_epochs + 1  # rows 0..num_epochs
            val_freq = int(cfg['training']['validation_frequency'])
            per_epoch_val = [float('nan')] * total_epochs
            per_epoch_train_cost = [float('nan')] * total_epochs
            per_epoch_train_loss = [float('nan')] * total_epochs

            # Parse captured legacy stdout for per-epoch mean loss and mean reward
            try:
                # Lines look like: "Epoch 5, mean loss: -18.207, mean reward: 14.230, time: 3.26"
                pat = re.compile(r"Epoch\s+(\d+)\s*,\s*mean loss:\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*mean reward:\s*([+-]?[0-9]*\.?[0-9]+)")
                for m in pat.finditer(legacy_stdout):
                    ep = int(m.group(1))
                    if 0 <= ep < total_epochs:
                        loss_v = float(m.group(2))
                        rew_v = float(m.group(3))
                        per_epoch_train_loss[ep] = loss_v
                        # Convert total reward (route length) to per-customer cost
                        per_epoch_train_cost[ep] = rew_v / float(n_customers)
            except Exception:
                pass

            # We'll also compute per-epoch train_cost via greedy evaluation for epochs lacking stdout entries

            # Evaluate checkpoints for train/val curves when CSV is missing or partial
            with torch.no_grad():
                # Evaluate validation at epochs [0, multiples of val_freq, num_epochs]
                eval_epochs = sorted(set([0] + [e for e in range(1, num_epochs + 1) if e % max(1, val_freq) == 0]))
                for ep in eval_epochs:
                    ck = os.path.join(ckpt_folder, str(ep), 'actor.pt')
                    if not os.path.exists(ck):
                        continue
                    # Load weights
                    try:
                        state = torch.load(ck, map_location='cpu')
                        legacy_model.load_state_dict(state)
                    except Exception:
                        continue
                    # Validation greedy cost per-customer
                    legacy_model.eval()
                    val_costs_acc = []
                    for batch in val_loader:
                        actions, _logp = legacy_model(batch, n_steps=n_steps, greedy=True, T=T)
                        depot = _torch.zeros(actions.size(0), 1, dtype=_torch.long)
                        full = _torch.cat([depot, actions, depot], dim=1)
                        ptr = batch.ptr if hasattr(batch, 'ptr') else None
                        for b in range(full.size(0)):
                            route = full[b].cpu().tolist()
                            if ptr is not None:
                                start = int(ptr[b].item()); end = int(ptr[b+1].item())
                                coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                            else:
                                coords = batch.x.view(-1, 2).cpu().numpy()
                            csum = 0.0
                            for i in range(len(route) - 1):
                                a = route[i]; c = route[i+1]
                                pa = coords[a]; pb = coords[c]
                                csum += float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
                            val_costs_acc.append(csum / float(n_customers))
                    if val_costs_acc:
                        per_epoch_val[ep] = float(np.mean(val_costs_acc))
                # Ensure final epoch val at num_epochs using the final trained model (no checkpoint needed)
                final_val_costs_acc = []
                legacy_model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        actions, _logp = legacy_model(batch, n_steps=n_steps, greedy=True, T=T)
                        depot = _torch.zeros(actions.size(0), 1, dtype=_torch.long)
                        full = _torch.cat([depot, actions, depot], dim=1)
                        ptr = batch.ptr if hasattr(batch, 'ptr') else None
                        for b in range(full.size(0)):
                            route = full[b].cpu().tolist()
                            if ptr is not None:
                                start = int(ptr[b].item()); end = int(ptr[b+1].item())
                                coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                            else:
                                coords = batch.x.view(-1, 2).cpu().numpy()
                            csum = 0.0
                            for i in range(len(route) - 1):
                                a = route[i]; c = route[i+1]
                                pa = coords[a]; pb = coords[c]
                                csum += float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
                            final_val_costs_acc.append(csum / float(n_customers))
                if final_val_costs_acc:
                    per_epoch_val[num_epochs] = float(np.mean(final_val_costs_acc))

                # Evaluate training cost per epoch for any epochs still missing train_cost
                for ep in range(0, num_epochs + 1):
                    if per_epoch_train_cost[ep] == per_epoch_train_cost[ep]:  # already set
                        continue
                    ck = os.path.join(ckpt_folder, str(ep), 'actor.pt')
                    if not os.path.exists(ck):
                        continue
                    try:
                        state = torch.load(ck, map_location='cpu')
                        legacy_model.load_state_dict(state)
                    except Exception:
                        continue
                    epoch_costs = []
                    for batch in train_loader:
                        actions, _logp = legacy_model(batch, n_steps=n_steps, greedy=True, T=T)
                        depot = _torch.zeros(actions.size(0), 1, dtype=_torch.long)
                        full = _torch.cat([depot, actions, depot], dim=1)
                        ptr = batch.ptr if hasattr(batch, 'ptr') else None
                        for b in range(full.size(0)):
                            route = full[b].cpu().tolist()
                            if ptr is not None:
                                start = int(ptr[b].item()); end = int(ptr[b+1].item())
                                coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                            else:
                                coords = batch.x.view(-1, 2).cpu().numpy()
                            csum = 0.0
                            for i in range(len(route) - 1):
                                a = route[i]; c = route[i+1]
                                pa = coords[a]; pb = coords[c]
                                csum += float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
                            epoch_costs.append(csum / float(n_customers))
                    if epoch_costs:
                        per_epoch_train_cost[ep] = float(np.mean(epoch_costs))

            # Ensure full coverage 0..num_epochs for train_loss/train_cost
            # Forward-fill train_loss if legacy logs miss some epochs, fallback to negative train_cost (proxy) then 0.0
            last_loss = None
            for ep in range(total_epochs):
                v = per_epoch_train_loss[ep]
                if v == v:  # not NaN
                    last_loss = v
                else:
                    if last_loss is not None:
                        per_epoch_train_loss[ep] = last_loss
                    else:
                        # fallback proxy: negative of train_cost if available, else 0.0
                        tc = per_epoch_train_cost[ep]
                        per_epoch_train_loss[ep] = (-tc if tc == tc else 0.0)
            # Back/forward-fill train_cost to ensure no gaps
            last_cost = None
            for ep in range(total_epochs):
                v = per_epoch_train_cost[ep]
                if v == v:
                    last_cost = v
                else:
                    if last_cost is not None:
                        per_epoch_train_cost[ep] = last_cost
            # Ensure epoch 0 is fully populated
            # If train_cost[0] NaN, use val_cost[0] as proxy or evaluate checkpoint 0
            if not (per_epoch_train_cost[0] == per_epoch_train_cost[0]):
                per_epoch_train_cost[0] = per_epoch_val[0] if per_epoch_val[0] == per_epoch_val[0] else 0.0
            # If train_loss[0] NaN, use parsed loss at 0 if available; else proxy -train_cost[0]
            if not (per_epoch_train_loss[0] == per_epoch_train_loss[0]):
                per_epoch_train_loss[0] = (-per_epoch_train_cost[0])
            # If val_cost[0] NaN, try evaluating earliest available checkpoint to fill it
            if not (per_epoch_val[0] == per_epoch_val[0]):
                with torch.no_grad():
                    for ep_try in range(0, num_epochs + 1):
                        ck = os.path.join(ckpt_folder, str(ep_try), 'actor.pt')
                        if not os.path.exists(ck):
                            continue
                        try:
                            state = torch.load(ck, map_location='cpu')
                            legacy_model.load_state_dict(state)
                        except Exception:
                            continue
                        vals = []
                        for batch in val_loader:
                            actions, _logp = legacy_model(batch, n_steps=n_steps, greedy=True, T=T)
                            depot = _torch.zeros(actions.size(0), 1, dtype=_torch.long)
                            full = _torch.cat([depot, actions, depot], dim=1)
                            ptr = batch.ptr if hasattr(batch, 'ptr') else None
                            for b in range(full.size(0)):
                                route = full[b].cpu().tolist()
                                if ptr is not None:
                                    start = int(ptr[b].item()); end = int(ptr[b+1].item())
                                    coords = batch.x[start:end].view(-1, 2).cpu().numpy()
                                else:
                                    coords = batch.x.view(-1, 2).cpu().numpy()
                                csum = 0.0
                                for i in range(len(route) - 1):
                                    a = route[i]; c = route[i+1]
                                    pa = coords[a]; pb = coords[c]
                                    csum += float(((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5)
                                vals.append(csum / float(n_customers))
                        if vals:
                            per_epoch_val[0] = float(np.mean(vals))
                            break

            # Build final series and artifacts
            final_val_candidates = [v for v in per_epoch_val[1:] if v == v]  # exclude epoch 0 for final selection
            final_val = float(final_val_candidates[-1]) if final_val_candidates else (per_epoch_val[num_epochs] if per_epoch_val[num_epochs] == per_epoch_val[num_epochs] else float('nan'))

            # Write legacy CSV history with per-epoch series (single source of truth)
            write_history_csv('GAT+RL (legacy)', {
                'train_losses': per_epoch_train_loss,
                'train_costs': per_epoch_train_cost,
                'val_costs': [v for v in per_epoch_val if v == v],
                'final_val_cost': final_val,
            }, cfg, base_dir, logger)

            # Save legacy checkpoint with richer results
            legacy_ckpt = os.path.join(base_dir, 'pytorch', 'model_gat_rl_legacy.pt')
            os.makedirs(os.path.dirname(legacy_ckpt), exist_ok=True)
            torch.save({'model_state_dict': legacy_model.state_dict(), 'model_name': 'GAT+RL (legacy)', 'config': cfg, 'results': {'train_losses': per_epoch_train_loss[1:], 'train_costs': per_epoch_train_cost[1:], 'val_costs': final_val_candidates, 'final_val_cost': final_val}, 'training_time': legacy_time}, legacy_ckpt)
            logger.info(f'💾 Saved legacy model checkpoint: {legacy_ckpt}')

            # Add to results for analysis
            results['GAT+RL (legacy)'] = {'history': {'train_losses': per_epoch_train_loss[1:], 'train_costs': per_epoch_train_cost[1:], 'val_costs': final_val_candidates, 'final_val_cost': final_val}}
            training_times['GAT+RL (legacy)'] = legacy_time
            models['GAT+RL (legacy)'] = legacy_model
        except Exception:
            # Dependencies missing or repo unavailable; already warned
            pass
        except Exception:
            # Dependencies missing or repo unavailable; already warned
            pass

    # Save artifacts after (possibly) adding legacy
    # CSVs are already written per-model above; just save checkpoints and analysis
    save_models_and_analysis(results, training_times, models, cfg, base_dir, logger)

    # Summary
    logger.info('\n📊 SUMMARY')
    for name, data in results.items():
        params = sum(p.numel() for p in models[name].parameters())
        final_val = data['history']['final_val_cost']
        logger.info(f'- {name}: params={params:,}, time={training_times[name]:.1f}s, final_val={final_val:.3f}/cust')


if __name__ == '__main__':
    main()

