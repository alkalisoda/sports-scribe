"""Test package for Sports Intelligence Layer.

Ensure package root is importable when tests are invoked via `python -m`.
"""

import os
import sys
from pathlib import Path

# Add project package root to sys.path if not present
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))