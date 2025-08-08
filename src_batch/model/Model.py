import os
import sys
# Ensure legacy repo is on path
LEGACY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GAT_RL'))
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)

# Import the legacy Model without modifying it
from model.Model import Model as _LegacyModel  # type: ignore

# Re-export with the expected name
class Model(_LegacyModel):
    pass

