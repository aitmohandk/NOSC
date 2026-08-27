"""
CPU/GPU-agnostic behaviour tests (require torch; run in the project env).
Verifies the selector works on CPU and that its index tensors follow the
batch's device, and that NOSC_DEVICE overrides selection.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap_no_heavy_deps  # noqa: F401
import types
_mv = types.ModuleType('contrib.multivar')
_mv.__path__ = [str(Path(__file__).resolve().parents[1] / 'contrib' / 'multivar')]
sys.modules.setdefault('contrib.multivar', _mv)

import torch  # noqa: E402
from contrib.multivar.device_utils import default_device  # noqa: E402


def test_default_device_env_override():
    os.environ['NOSC_DEVICE'] = 'cpu'
    default_device.cache_clear()
    assert default_device().type == 'cpu'
    del os.environ['NOSC_DEVICE']
    default_device.cache_clear()


def test_default_device_matches_hardware():
    default_device.cache_clear()
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert default_device().type == expected


def _make_selector():
    from contrib.multivar.multivar_utils import MultivarBatchSelector
    # minimal multivar_info: 2 input vars (idx 0,1), 1 output var (idx 2)
    info = dict(
        var_names=['ssh_obs', 'sst_in', 'zos_tgt'],
        full_input_idx=[], prior_input_idx=[0, 1], full_output_idx=[2],
        state_obs_channels=[], state_obs_input_idx=[0],
    )
    try:
        return MultivarBatchSelector(info)
    except TypeError:
        sel = MultivarBatchSelector.__new__(MultivarBatchSelector)
        for k in ('full_input_idx', 'prior_input_idx', 'full_output_idx',
                  'state_obs_channels', 'state_obs_input_idx'):
            setattr(sel, k, torch.tensor(info[k], dtype=torch.int64))
        sel.var_names = info['var_names']
        return sel


def test_selector_runs_on_cpu_and_follows_batch_device():
    sel = _make_selector()
    batch = torch.randn(2, 3, 4, 8, 8)  # (B, C=3 vars, T=4, H, W)
    out = sel.multivar_prior_input(batch)
    assert out.device == batch.device
    assert out.shape[0] == 2
    tgt = sel.multivar_full_output(batch)
    assert tgt.device == batch.device


if __name__ == '__main__':
    for fn_name in list(globals()):
        if fn_name.startswith('test_'):
            print(f"{fn_name} ...", end=' ', flush=True)
            globals()[fn_name]()
            print("OK")
    print("All device-agnostic tests passed.")
