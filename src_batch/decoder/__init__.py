import os
import sys
LEGACY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GAT_RL'))
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)
from decoder import *  # type: ignore

