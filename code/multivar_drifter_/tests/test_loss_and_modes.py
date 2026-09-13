"""
Numpy/pure-python tests for the loss-grouping and vertical-EOF logic
(runnable without torch/xarray, like test_synthetic_obs.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap_no_heavy_deps  # noqa: F401
import types
for pkg, sub in [('contrib.multivar', 'multivar')]:
    mod = types.ModuleType(pkg)
    mod.__path__ = [str(Path(__file__).resolve().parents[1] / 'contrib' / sub)]
    sys.modules.setdefault(pkg, mod)

from contrib.multivar.loss_grouping import combine_grouped_losses  # noqa: E402
from contrib.multivar.vertical_modes import eofs_from_profiles  # noqa: E402


def test_flat_sum_matches_historical_behaviour():
    losses = [1.0, 2.0, 3.0]
    total, groups = combine_grouped_losses(losses, ['a', 'b', 'c'], mode='flat_sum')
    assert total == 6.0 and groups == {'all': 6.0}


def test_group_mean_equalizes_physical_quantities():
    # 1 SSH channel vs 3 temperature channels: with group_mean, each family
    # contributes its mean -> SSH is no longer drowned 1-to-3.
    names = ['zos', 't1', 't2', 't3']
    groups = {'zos': 'ssh', 't1': 'temp', 't2': 'temp', 't3': 'temp'}
    total, per_group = combine_grouped_losses([4.0, 1.0, 2.0, 3.0], names,
                                              loss_groups=groups, mode='group_mean')
    assert per_group['ssh'] == 4.0
    assert per_group['temp'] == 2.0          # mean of 1,2,3
    assert total == 6.0


def test_group_weights_apply():
    names = ['a', 'b']
    groups = {'a': 'g1', 'b': 'g2'}
    total, _ = combine_grouped_losses([1.0, 1.0], names, loss_groups=groups,
                                       mode='group_mean', group_weights={'g2': 3.0})
    assert total == 4.0


def test_eofs_orthonormal_ordered_and_reconstructive():
    rng = np.random.default_rng(0)
    # synthetic profiles = 3 smooth modes + noise, 500 samples x 21 levels
    z = np.linspace(0, 1, 21)
    basis = np.stack([np.ones_like(z), z - z.mean(), (z - 0.5) ** 2 - ((z - 0.5) ** 2).mean()])
    coeffs = rng.normal(size=(500, 3)) * np.array([3.0, 2.0, 1.0])
    profiles = coeffs @ basis + 0.05 * rng.normal(size=(500, 21))

    comp, evr = eofs_from_profiles(profiles, n_modes=5)
    assert comp.shape == (5, 21)
    assert np.allclose(comp @ comp.T, np.eye(5), atol=1e-8)          # orthonormal
    assert np.all(np.diff(evr) <= 1e-12)                             # decreasing variance
    assert evr[:3].sum() > 0.95                                      # 3 true modes dominate

    # reconstruction error decreases with more modes
    anom = profiles - profiles.mean(axis=0)
    def rec_err(k):
        p = comp[:k]
        return np.mean((anom - (anom @ p.T) @ p) ** 2)
    errs = [rec_err(k) for k in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]
    assert rec_err(3) < 0.01 * np.mean(anom ** 2)                    # 3 modes ~ perfect


def test_eofs_reject_underdetermined_input():
    try:
        eofs_from_profiles(np.random.default_rng(0).normal(size=(10, 21)), 5)
    except ValueError:
        return
    raise AssertionError("accepted fewer samples than levels")


def test_mask_sequential_and_cycle_modes():
    # torch-free re-import of the pure-numpy function from data.py's source
    import re
    src = (Path(__file__).resolve().parents[1] / 'contrib' / 'data_loading' / 'data.py').read_text()
    m = re.search(r"def mask_input_sequential.*?(?=\ndef |\nclass )", src, re.S)
    ns = {'np': np}
    exec(m.group(0), ns)
    f = ns['mask_input_sequential']

    da = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)
    masks = np.full((2, 2, 2), np.nan, dtype=np.float32)
    masks[0, 0, 0] = 1.0   # day-type A observes cell (0,0)
    masks[1, 1, 1] = 1.0   # day-type B observes cell (1,1)

    # sequential with too few masks -> explicit error
    try:
        f(da, masks)
        raise AssertionError('sequential accepted a short mask set')
    except ValueError:
        pass

    # cycle mode: day t uses mask[t % 2]
    out = f(da, masks, mode='cycle')
    assert np.isfinite(out[0, 0, 0]) and np.isnan(out[0, 1, 1])
    assert np.isfinite(out[1, 1, 1]) and np.isnan(out[1, 0, 0])
    assert np.isfinite(out[4, 0, 0])            # 4 % 2 == 0 -> type A again
    assert np.isnan(out[2, 0, 1])               # never observed anywhere

    # sequential with enough masks: day-aligned, no recycling
    masks5 = np.concatenate([masks, masks, masks[:1]], axis=0)
    out5 = f(da, masks5)
    assert np.isfinite(out5[4, 0, 0]) and np.isnan(out5[4, 1, 1])


if __name__ == '__main__':
    for fn_name in list(globals()):
        if fn_name.startswith('test_'):
            print(f"{fn_name} ...", end=' ', flush=True)
            globals()[fn_name]()
            print("OK")
    print("All loss/modes tests passed.")
