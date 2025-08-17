import os
import sys
# Ensure legacy repo on path
LEGACY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GAT_RL'))
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)
# Re-export legacy package
from encoder import *  # type: ignore

