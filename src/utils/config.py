import os
import yaml
from typing import Dict, Any


def _deep_merge_dict(a: dict, b: dict) -> dict:
    """Recursively merge dict b into dict a (returns a).
    - Scalars/lists in b overwrite a
    - Dicts are merged recursively
    """
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge_dict(a[k], v)
        else:
            a[k] = v
    return a


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return x


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and normalize configuration by deep-merging configs/default.yaml with config_path.

    Guarantees:
    - All parameters originate from YAML files (no hidden defaults in code)
    - Required sections/keys validated
    - Types normalized (ints/floats)
    - A flattened view is added for convenience: num_customers, capacity, coord_range,
      demand_range, num_instances, batch_size, num_epochs, learning_rate, hidden_dim,
      num_heads, num_layers, entropy/temperature schedule, grad_clip, etc.
    """
    default_path = os.path.join('configs', 'default.yaml')
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default config not found at {default_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(default_path, 'r') as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(config_path, 'r') as f:
        override_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge_dict(base_cfg, override_cfg)

    # Validate presence of required sections
    if 'problem' not in cfg:
        raise ValueError("Missing 'problem' section in configuration file")
    if 'training' not in cfg:
        raise ValueError("Missing 'training' section in configuration file")
    if 'model' not in cfg:
        raise ValueError("Missing 'model' section in configuration file")

    # Required keys
    required_problem = ['num_customers', 'vehicle_capacity', 'coord_range', 'demand_range']
    required_training = ['num_instances', 'batch_size', 'num_epochs', 'learning_rate']
    required_model = ['input_dim', 'hidden_dim', 'num_heads', 'num_layers', 'transformer_dropout', 'feedforward_multiplier', 'edge_embedding_divisor']

    for k in required_problem:
        if k not in cfg['problem']:
            raise ValueError(f"Missing problem.{k} in configuration")
    for k in required_training:
        if k not in cfg['training']:
            raise ValueError(f"Missing training.{k} in configuration")
    for k in required_model:
        if k not in cfg['model']:
            raise ValueError(f"Missing model.{k} in configuration")

    # Type normalization for nested sections
    p = cfg['problem']
    t = cfg['training']
    m = cfg['model']

    p['num_customers'] = _to_int(p['num_customers'])
    p['vehicle_capacity'] = _to_int(p['vehicle_capacity'])
    p['coord_range'] = _to_int(p['coord_range'])
    # demand_range is a 2-list
    if isinstance(p.get('demand_range'), (list, tuple)) and len(p['demand_range']) == 2:
        p['demand_range'] = [_to_int(p['demand_range'][0]), _to_int(p['demand_range'][1])]

    t['num_instances'] = _to_int(t['num_instances'])
    t['batch_size'] = _to_int(t['batch_size'])
    t['num_epochs'] = _to_int(t['num_epochs'])
    t['learning_rate'] = _to_float(t['learning_rate'])
    if 'train_val_split' in t:
        t['train_val_split'] = float(t['train_val_split'])

    m['input_dim'] = _to_int(m['input_dim'])
    m['hidden_dim'] = _to_int(m['hidden_dim'])
    m['num_heads'] = _to_int(m['num_heads'])
    m['num_layers'] = _to_int(m['num_layers'])
    m['transformer_dropout'] = float(m['transformer_dropout'])
    m['feedforward_multiplier'] = _to_int(m['feedforward_multiplier'])
    m['edge_embedding_divisor'] = _to_int(m['edge_embedding_divisor'])

    # training_advanced
    ta = cfg.get('training_advanced', {})
    if ta:
        if 'gradient_clip_norm' in ta:
            ta['gradient_clip_norm'] = _to_float(ta['gradient_clip_norm'])
        if 'warmup_epochs' in ta:
            ta['warmup_epochs'] = _to_int(ta['warmup_epochs'])
        if 'min_lr' in ta:
            ta['min_lr'] = _to_float(ta['min_lr'])
        if 'entropy_coef' in ta:
            ta['entropy_coef'] = _to_float(ta['entropy_coef'])
        if 'entropy_min' in ta:
            ta['entropy_min'] = _to_float(ta['entropy_min'])
        if 'temp_start' in ta:
            ta['temp_start'] = _to_float(ta['temp_start'])
        if 'temp_min' in ta:
            ta['temp_min'] = _to_float(ta['temp_min'])
        if 'legacy_gat' in ta:
            lg = ta['legacy_gat']
            if 'learning_rate' in lg:
                lg['learning_rate'] = _to_float(lg['learning_rate'])
            if 'temperature' in lg:
                lg['temperature'] = _to_float(lg['temperature'])
            if 'max_steps_multiplier' in lg:
                lg['max_steps_multiplier'] = _to_int(lg['max_steps_multiplier'])

    # inference
    inf = cfg.get('inference', {})
    if inf:
        if 'default_temperature' in inf:
            inf['default_temperature'] = _to_float(inf['default_temperature'])
        if 'max_steps_multiplier' in inf:
            inf['max_steps_multiplier'] = _to_int(inf['max_steps_multiplier'])
        if 'attention_temperature_scaling' in inf:
            inf['attention_temperature_scaling'] = _to_float(inf['attention_temperature_scaling'])
        if 'log_prob_epsilon' in inf:
            inf['log_prob_epsilon'] = _to_float(inf['log_prob_epsilon'])
        if 'masked_score_value' in inf:
            inf['masked_score_value'] = _to_float(inf['masked_score_value'])

    # Flattened convenience keys (no hidden defaults; all sourced from YAML)
    cfg.update({
        'num_customers': p['num_customers'],
        'capacity': p['vehicle_capacity'],
        'coord_range': p['coord_range'],
        'demand_range': p['demand_range'],
        'num_instances': t['num_instances'],
        'batch_size': t['batch_size'],
        'num_epochs': t['num_epochs'],
        'learning_rate': t['learning_rate'],
        'hidden_dim': m['hidden_dim'],
        'num_heads': m['num_heads'],
        'num_layers': m['num_layers'],
    })

    if ta:
        cfg.update({
            'grad_clip': ta.get('gradient_clip_norm'),
            'warmup_epochs': ta.get('warmup_epochs'),
            'min_lr': ta.get('min_lr'),
            'entropy_coef': ta.get('entropy_coef'),
            'entropy_min': ta.get('entropy_min'),
            'temp_start': ta.get('temp_start'),
            'temp_min': ta.get('temp_min'),
        })

    return cfg

