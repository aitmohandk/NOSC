"""
Test bootstrap for environments without the full conda env: registers the
'contrib' and 'contrib.synthetic_obs' packages WITHOUT executing
contrib/__init__.py (which chain-imports hydra/torch-dependent subpackages).
Harmless in the full env; required for the pure-numpy tests to run anywhere.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for pkg, path in [('contrib', ROOT / 'contrib'),
                  ('contrib.synthetic_obs', ROOT / 'contrib' / 'synthetic_obs')]:
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(path)]
        sys.modules[pkg] = mod
