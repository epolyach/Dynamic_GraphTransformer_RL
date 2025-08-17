# Re-export legacy modules so that imports like src_batch.encoder.GAT_Encoder work.
# We insert the legacy repo root into sys.path at runtime when this shim is imported.
import os
import sys
from importlib import import_module

LEGACY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GAT_RL'))
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)

# Directly load legacy packages into our namespace using file paths
from ._legacy_loader import load_legacy_module

# Map of logical package names to their legacy __init__.py paths
LEGACY_MAP = {
    'encoder': os.path.join(LEGACY_ROOT, 'encoder', '__init__.py'),
    'decoder': os.path.join(LEGACY_ROOT, 'decoder', '__init__.py'),
    'model': os.path.join(LEGACY_ROOT, 'model', '__init__.py'),
    'RL': os.path.join(LEGACY_ROOT, 'RL', '__init__.py'),
    'instance_creator': os.path.join(LEGACY_ROOT, 'instance_creator', '__init__.py'),
}

for name, path in LEGACY_MAP.items():
    if os.path.exists(path):
        load_legacy_module(name, path)
    else:
        # If a package is missing, we proceed; only needed ones will be imported by clients
        pass

