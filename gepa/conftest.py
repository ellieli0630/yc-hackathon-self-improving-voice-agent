"""Put this directory on sys.path so the tests' flat imports
(`from judge import ...`, `from cekura_score import ...`, etc.) resolve as
top-level modules, independent of how pytest is invoked.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
