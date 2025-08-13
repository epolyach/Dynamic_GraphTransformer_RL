import os
import json
from typing import Dict, Any

import torch


def ensure_dirs(base_dir: str, subdirs: Dict[str, str]) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    paths = {}
    for key, name in subdirs.items():
        p = os.path.join(base_dir, name)
        os.makedirs(p, exist_ok=True)
        paths[key] = p
    return paths


def save_model_checkpoint(model_name: str, model: torch.nn.Module, base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    fname = f"model_{model_name}.pt"
    path = os.path.join(base_dir, fname)
    torch.save(model.state_dict(), path)
    return path


def save_analysis_blob(blob: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

