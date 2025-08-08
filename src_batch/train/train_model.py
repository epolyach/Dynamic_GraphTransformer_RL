# Forwarder so imports like `from src_batch.train.train_model import train` resolve
# to the legacy repo's training script without altering legacy code.
import os
import sys

LEGACY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GAT_RL'))
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)

# Import the legacy training module in a way that preserves package context
# so "from ..RL ..." relative imports inside legacy code can resolve.
import types
import importlib

# Create a synthetic top-level package named "GAT_RL" that points to LEGACY_ROOT
pkg_name = 'GAT_RL'
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [LEGACY_ROOT]
    sys.modules[pkg_name] = pkg

# Now import the legacy training module as a submodule of GAT_RL
legacy_mod = importlib.import_module('GAT_RL.train.train_model')
train = getattr(legacy_mod, 'train')

