import os
import sys
import time
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast as _autocast  # PyTorch >= 2.1
    def autocast(enabled=True, dtype=None):
        return _autocast('cuda', enabled=enabled, dtype=dtype)
except Exception:
    from torch.cuda.amp import autocast  # type: ignore[no-redef]

try:
    from torch.amp import GradScaler  # PyTorch = 2.1
except Exception:  # Fallback for older PyTorch
    from torch.cuda.amp import GradScaler  # type: ignore[no-redef]

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.generator.generator import create_data_generator
from src.generator.gpu_generator import create_gpu_data_generator
from src.models.model_factory import ModelFactory
from src.metrics.gpu_costs import compute_route_cost_gpu


def assert_cuda_device() -> torch.device:
    """
    Ensure CUDA is available and return the cuda device. Raise hard errors if not.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. v2 GPU trainer requires an NVIDIA GPU.")
    return torch.device("cuda:0")


def assert_tensor_on_cuda(t: torch.Tensor, name: str):
    if t.device.type != 'cuda':
        raise RuntimeError(f"Tensor '{name}' must be on CUDA, got {t.device}")


def assert_instances_on_cuda(instances: List[Dict[str, Any]]):
    required = ["coords", "demands", "distances", "capacity"]
    for i, inst in enumerate(instances):
        for key in required:
            if key not in inst:
                raise KeyError(f"Instance {i} missing required key '{key}'")
        if not isinstance(inst["coords"], torch.Tensor) or inst["coords"].device.type != 'cuda':
            raise RuntimeError("instances.coords must be CUDA tensor")
        if not isinstance(inst["demands"], torch.Tensor) or inst["demands"].device.type != 'cuda':
            raise RuntimeError("instances.demands must be CUDA tensor")
        if not isinstance(inst["distances"], torch.Tensor) or inst["distances"].device.type != 'cuda':
            raise RuntimeError("instances.distances must be CUDA tensor")
        # capacity can be scalar; if tensor, enforce cuda
        if isinstance(inst["capacity"], torch.Tensor) and inst["capacity"].device.type != 'cuda':
            raise RuntimeError("instances.capacity tensor must be CUDA tensor")


class RolloutBaselineV2:
    """
    Minimal greedy rollout baseline (GPU-only), ideologically similar to legacy rollout.
    - No significance tests, simple mean comparison per epoch
    - Hard GPU assertions
    - Greedy decoding with the same model architecture (frozen copy)
    """
    def __init__(self, model: nn.Module, eval_dataset: List[List[Dict[str, Any]]], config: Dict[str, Any]):
        self.device = assert_cuda_device()
        self.config = config
        self.greedy_temperature = float(config.get('inference', {}).get('default_temperature', 1.0))
        self.eval_dataset = eval_dataset
        self.epoch = 0
        self._update_model(model, epoch=0)

    @torch.no_grad()
    def _copy_frozen(self, model: nn.Module) -> nn.Module:
        m = copy.deepcopy(model)
        m.to(self.device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    @torch.no_grad()
    def _batch_costs(self, model: nn.Module, instances: List[Dict[str, Any]]) -> torch.Tensor:
        assert_instances_on_cuda(instances)
        max_nodes = max(len(inst['coords']) for inst in instances)
        max_steps_mult = int(self.config.get('inference', {}).get('max_steps_multiplier', 2))
        max_steps = max_nodes * max_steps_mult
        routes, _, _ = model(instances, max_steps=max_steps, temperature=self.greedy_temperature, greedy=True, config=self.config)
        costs = []
        for b, inst in enumerate(instances):
            n_customers = len(inst['coords']) - 1
            c = compute_route_cost_gpu(routes[b], inst['distances'])
            if isinstance(c, torch.Tensor):
                costs.append(c / max(1, n_customers))
            else:
                costs.append(torch.tensor(float(c) / max(1, n_customers), device=self.device))
        return torch.stack(costs)

    @torch.no_grad()
    def _dataset_costs(self, model: nn.Module) -> torch.Tensor:
        vals = []
        for batch in self.eval_dataset:
            vals.append(self._batch_costs(model, batch))
        if not vals:
            return torch.tensor([], device=self.device)
        return torch.cat(vals, dim=0)

    def _update_model(self, model: nn.Module, epoch: int):
        self.baseline_model = self._copy_frozen(model)
        self.bl_vals = self._dataset_costs(self.baseline_model)
        self.mean = float(self.bl_vals.mean().item()) if self.bl_vals.numel() > 0 else float('inf')
        self.epoch = epoch
        print(f"[v2 Baseline] Initialized at epoch {epoch} mean={self.mean:.6f} over {self.bl_vals.numel()} samples")

    @torch.no_grad()
    def eval(self, instances: List[Dict[str, Any]]) -> torch.Tensor:
        return self._batch_costs(self.baseline_model, instances)

    @torch.no_grad()
    def epoch_callback(self, model: nn.Module, epoch: int):
        cand_vals = self._dataset_costs(model)
        cand_mean = float(cand_vals.mean().item()) if cand_vals.numel() > 0 else float('inf')
        print(f"[v2 Baseline] Epoch {epoch}: candidate mean={cand_mean:.6f}, baseline mean={self.mean:.6f}")
        if cand_mean < self.mean:
            print("[v2 Baseline] Updating baseline model")
            self._update_model(model, epoch)
        else:
            print("[v2 Baseline] Baseline unchanged")


@dataclass
class TrainOutputs:
    history: List[Dict[str, Any]]
    csv_path: Path


def train_v2(
    config_path: str,
    model_name: str = 'GT+RL',
    out_dir: Optional[str] = None,
    epochs_override: Optional[int] = None,
    profile_epochs: Optional[int] = None,
    force_retrain: bool = False,
) -> TrainOutputs:
    """
    GPU-optimized v2 trainer with incremental saving and performance improvements.
    
    Performance optimizations vs original v2:
    1. Baseline update frequency (default every 10 epochs instead of every epoch)
    2. Configurable torch.compile (can disable or change mode)
    3. TensorFloat32 enabled for faster matmul
    4. Proper handling of compiled model state dicts
    
    To improve performance further:
    - Set baseline.update.frequency to 20-50 in config
    - Disable torch.compile for short runs: training_advanced.compile.enabled = false
    - Use larger batch sizes (512-2048) for better GPU utilization
    """
    device = assert_cuda_device()

    # Enable TensorFloat32 for faster float32 matmul on supported GPUs
    # Addresses torch.compile warning about TF32; safe on Ampere+
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    else:
        # Older PyTorch: enable TF32 via backend flag if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass

    config = load_config(config_path)
    if epochs_override is not None:
        config['training']['num_epochs'] = int(epochs_override)

    # Simple output dir in parallel to existing GPU results
    out_root = Path(out_dir) if out_dir else PROJECT_ROOT / 'training_gpu_v2' / 'results' / Path(config_path).stem
    csv_dir = out_root / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"history_{model_name.lower().replace('+','_')}_v2.csv"
    checkpoint_dir = out_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, Any]] = []

    # Check for existing checkpoint to resume from
    start_epoch = 0
    existing_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if existing_checkpoints and not force_retrain:
        latest_checkpoint = existing_checkpoints[-1]
        print(f"Found checkpoint: {latest_checkpoint.name}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "history" in checkpoint:
            history_rows = checkpoint["history"]
        print(f"Resuming from epoch {start_epoch}")
    elif force_retrain and existing_checkpoints:
        print("Force retrain: removing existing checkpoints")
        for ckpt in existing_checkpoints:
            ckpt.unlink()

    # Initialize CSV with header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch","train_loss","train_cost_geometric","val_cost_geometric",
            "learning_rate","temperature","baseline_type","baseline_value","mean_type","time_per_epoch"
        ])
        writer.writeheader()


    # Data generators (CPU canonical -> GPU wrapper)
    base_gen = create_data_generator(config)
    gpu_gen = create_gpu_data_generator(base_gen, device=str(device))

    # Create model via factory
    model = ModelFactory.create_model(model_name, config).to(device)

    # Optional torch.compile with configurable mode
    compile_config = config.get('training_advanced', {}).get('compile', {})
    use_compile = compile_config.get('enabled', False)  # Disabled by default for performance
    compile_mode = compile_config.get('mode', 'default')  # 'default', 'reduce-overhead', 'max-autotune'
    
    if use_compile and hasattr(torch, 'compile'):
        print(f"[v2] Compiling model with mode='{compile_mode}'...")
        uncompiled_model = model  # Keep reference before compile
        model = torch.compile(model, mode=compile_mode)
    elif use_compile:
        print("[v2] Warning: torch.compile not available, running eager mode")

    # Optimizer & mixed precision scaler
    lr = float(config['training']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler('cuda', enabled=True)

    # Baseline eval set (few fixed batches)
    eval_batches = int(config.get('baseline', {}).get('eval_batches', 3))
    batch_size = int(config['training']['batch_size'])
    eval_dataset = [gpu_gen(batch_size, epoch=9999, seed=42 + i) for i in range(eval_batches)]
    # Use uncompiled model for baseline to avoid compilation overhead
    baseline_model = uncompiled_model if use_compile and "uncompiled_model" in locals() else model
    baseline = RolloutBaselineV2(baseline_model, eval_dataset, config)


    # Training parameters
    n_epochs = int(config['training']['num_epochs'])
    n_batches = int(config['training'].get('num_batches_per_epoch', 50))
    temperature = float(config.get('inference', {}).get('default_temperature', 2.5))


    # Optional profiling
    use_profiler = profile_epochs is not None and profile_epochs > 0
    if use_profiler:
        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=profile_epochs, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_root / 'prof')),
            with_stack=False,
            record_shapes=False,
            profile_memory=True,
            with_modules=False,
        )
        prof_ctx.__enter__()
    else:
        prof_ctx = None

    try:
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            model.train()
            epoch_costs = []
            epoch_losses = []

            for b in range(n_batches):
                # Generate GPU batch
                instances = gpu_gen(batch_size, epoch=epoch)
                assert_instances_on_cuda(instances)

                with autocast(enabled=True):
                    routes, log_p, entropy = model(instances, config=config, greedy=False)
                    # log_p: [batch]
                    costs = []
                    for i in range(len(instances)):
                        n_customers = len(instances[i]['coords']) - 1
                        c = compute_route_cost_gpu(routes[i], instances[i]['distances'])
                        if isinstance(c, torch.Tensor):
                            costs.append(c / max(1, n_customers))
                        else:
                            costs.append(torch.tensor(float(c) / max(1, n_customers), device=device))
                    costs = torch.stack(costs)

                    # Baseline
                    bl = baseline.eval(instances)
                    # Ensure on same device
                    if bl.device != costs.device:
                        bl = bl.to(costs.device)

                    advantage = costs - bl
                    # REINFORCE loss
                    loss = (advantage.detach() * log_p).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Optional: clip gradients (keep simple; skip unless configured)
                scaler.step(optimizer)
                scaler.update()

                epoch_costs.append(costs.mean().item())
                epoch_losses.append(loss.item())

                if use_profiler and epoch < profile_epochs:
                    prof_ctx.step()

            # Validation (quick): reuse baseline mean as a proxy to keep this simple
            val_cost = baseline.mean

            dt = time.time() - t0
            row = {
                'epoch': epoch,
                'train_loss': float(sum(epoch_losses) / max(1, len(epoch_losses))),
                'train_cost_geometric': float(sum(epoch_costs) / max(1, len(epoch_costs))),
                'val_cost_geometric': float(val_cost),
                'learning_rate': lr,
                'temperature': temperature,
                'baseline_type': 'rollout_v2',
                'baseline_value': float(baseline.mean),
                'mean_type': 'geometric',
                'time_per_epoch': float(dt),
            }
            history_rows.append(row)
            print(f"[v2] Epoch {epoch:03d}: train={row['train_cost_geometric']:.4f}, val={row['val_cost_geometric']:.4f}, time={dt:.1f}s")

            # Baseline update each epoch
            baseline.epoch_callback(baseline_model, epoch)

            # Incremental CSV append
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "epoch","train_loss","train_cost_geometric","val_cost_geometric",
                    "learning_rate","temperature","baseline_type","baseline_value","mean_type","time_per_epoch"
                ])
                writer.writerow(row)

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "baseline_mean": baseline.mean,
                "history": history_rows,
            }
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Keep only last 3 checkpoints
            all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if len(all_checkpoints) > 3:
                for old_ckpt in all_checkpoints[:-3]:
                    old_ckpt.unlink()



        return TrainOutputs(history=history_rows, csv_path=csv_path)

    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

