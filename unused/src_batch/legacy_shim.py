# Re-export legacy modules so that imports like src_batch.encoder.GAT_Encoder work.
# We insert the legacy repo root into sys.path at runtime when this shim is imported.
import os
import sys

# Resolve the legacy GAT_RL root robustly:
# 1) Honor explicit environment variable LEGACY_GAT_RL_DIR if set
# 2) Try common relative locations based on this repository layout
#    - repo_root/../GAT_RL (preferred; user keeps GAT_RL next to this repo)
#    - repo_root/GAT_RL
#    - $HOME/GAT_RL
CANDIDATES = []
env_path = os.environ.get('LEGACY_GAT_RL_DIR')
if env_path:
    CANDIDATES.append(env_path)

here = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(here, '..'))  # Dynamic_GraphTransformer_RL
CANDIDATES.append(os.path.abspath(os.path.join(repo_root, '..', 'GAT_RL')))
CANDIDATES.append(os.path.abspath(os.path.join(repo_root, 'GAT_RL')))
CANDIDATES.append(os.path.abspath(os.path.join(os.path.expanduser('~'), 'GAT_RL')))

LEGACY_ROOT = None
for p in CANDIDATES:
    if p and os.path.isdir(p) and os.path.exists(os.path.join(p, '__init__.py')) or True:
        # Accept directory if it exists and contains expected subpackages
        if os.path.isdir(os.path.join(p, 'model')) or os.path.isdir(os.path.join(p, 'encoder')):
            LEGACY_ROOT = p
            break

if not LEGACY_ROOT:
    # Fallback to original heuristic (may not exist)
    LEGACY_ROOT = os.path.abspath(os.path.join(here, '..', '..', '..', 'GAT_RL'))

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

