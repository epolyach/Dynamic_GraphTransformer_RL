import copy
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ttest_rel

from src.metrics.costs import compute_route_cost
from src.eval.validation import validate_route


class RolloutBaseline:
    """
    Greedy rollout baseline, adapted from Kool et al. (2019).

    - Holds a frozen copy of the policy network (baseline model)
    - Evaluates it greedily on batches to produce per-instance baseline costs
    - Updates the baseline model when the current policy statistically outperforms it
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eval_dataset: List[List[Dict[str, Any]]],
        config: Dict[str, Any],
        logger_print: Callable[[str], None] = print,
    ) -> None:
        self.logger_print = logger_print
        self.config = config
        self.device = torch.device(config.get('experiment', {}).get('device', 'cpu'))

        # Greedy temperature used for evaluation/rollouts
        self.greedy_temperature: float = float(
            config.get('baseline', {}).get('greedy_temperature',
                   config.get('inference', {}).get('default_temperature', 0.1))
        )

        # Baseline update configuration
        upd_cfg = config.get('baseline', {}).get('update', {})
        self.update_enabled: bool = bool(upd_cfg.get('enabled', True))
        self.update_frequency: int = int(upd_cfg.get('frequency', 1))
        self.use_significance_test: bool = bool(upd_cfg.get('significance_test', True))
        self.significance_p_value: float = float(upd_cfg.get('p_value', 0.05))

        # Fixed evaluation dataset (list of batches of instances)
        self.eval_dataset: List[List[Dict[str, Any]]] = eval_dataset

        # Initialize baseline model and stats
        self.epoch: int = 0
        self._update_model(model, epoch=0)

    def _deepcopy_model(self, model: torch.nn.Module) -> torch.nn.Module:
        clone = copy.deepcopy(model)
        clone.to(self.device)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        return clone

    def _compute_batch_costs(self, model: torch.nn.Module, instances: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute normalized per-instance costs for a batch using greedy decoding.
        """
        if len(instances) == 0:
            return np.zeros((0,), dtype=np.float32)

        with torch.no_grad():
            max_nodes = max(len(inst['coords']) for inst in instances)
            max_steps_mult = int(self.config.get('inference', {}).get('max_steps_multiplier', 2))
            max_steps = max_nodes * max_steps_mult
            routes, _, _ = model(
                instances,
                max_steps=max_steps,
                temperature=self.greedy_temperature,
                greedy=True,
                config=self.config,
            )

        costs: List[float] = []
        for b, inst in enumerate(instances):
            n_customers = len(inst['coords']) - 1
            r = routes[b]
            # Validate strictly; raise if invalid to catch issues early
            validate_route(r, n_customers, model_name="Baseline-Greedy", instance=inst)
            c = compute_route_cost(r, inst['distances'])
            c_norm = c / max(1, n_customers)
            costs.append(float(c_norm))
        return np.asarray(costs, dtype=np.float32)

    def _compute_dataset_costs(self, model: torch.nn.Module) -> np.ndarray:
        vals: List[np.ndarray] = []
        for batch in self.eval_dataset:
            vals.append(self._compute_batch_costs(model, batch))
        if len(vals) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(vals, axis=0)

    def _update_model(self, model: torch.nn.Module, epoch: int) -> None:
        self.baseline_model = self._deepcopy_model(model)
        self.bl_vals = self._compute_dataset_costs(self.baseline_model)
        self.mean = float(self.bl_vals.mean()) if self.bl_vals.size > 0 else float('inf')
        self.epoch = epoch
        self.logger_print(
            f"[RolloutBaseline] Initialized at epoch {epoch} with mean={self.mean:.6f} "
            f"over {self.bl_vals.size} eval samples"
        )

    def eval_batch(self, instances: List[Dict[str, Any]]) -> torch.Tensor:
        """Return per-instance baseline costs as a tensor (CPU)."""
        costs = self._compute_batch_costs(self.baseline_model, instances)
        return torch.tensor(costs, dtype=torch.float32)

    def epoch_callback(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Evaluate candidate policy on the fixed eval set; if mean improvement is
        significant (or simply lower if significance disabled), update baseline.
        """
        if not self.update_enabled:
            return
        if (epoch % self.update_frequency) != 0:
            return

        self.logger_print("[RolloutBaseline] Evaluating candidate on eval dataset...")
        candidate_model = self._deepcopy_model(model)
        candidate_vals = self._compute_dataset_costs(candidate_model)
        candidate_mean = float(candidate_vals.mean()) if candidate_vals.size > 0 else float('inf')

        self.logger_print(
            f"[RolloutBaseline] Epoch {epoch}: candidate mean={candidate_mean:.6f}, "
            f"baseline (epoch {self.epoch}) mean={self.mean:.6f}, diff={candidate_mean - self.mean:.6f}"
        )

        if candidate_mean < self.mean:
            should_update = True
            if self.use_significance_test and self.bl_vals.size > 0 and candidate_vals.size == self.bl_vals.size:
                try:
                    t, p_two_sided = ttest_rel(candidate_vals, self.bl_vals)
                    # one-sided test: candidate < baseline => t should be negative
                    p_one_sided = float(p_two_sided / 2.0)
                    self.logger_print(f"[RolloutBaseline] t={float(t):.4f}, one-sided p={p_one_sided:.6f}")
                    should_update = (t < 0) and (p_one_sided < self.significance_p_value)
                except Exception as e:
                    self.logger_print(f"[RolloutBaseline] Significance test failed: {e}. Proceeding with mean check only.")
                    should_update = True
            if should_update:
                self.logger_print("[RolloutBaseline] Updating baseline model to candidate.")
                self._update_model(model, epoch)
            else:
                self.logger_print("[RolloutBaseline] Candidate not significantly better; baseline unchanged.")
        else:
            self.logger_print("[RolloutBaseline] Candidate worse; baseline unchanged.")
