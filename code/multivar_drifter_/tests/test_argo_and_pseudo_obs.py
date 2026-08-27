"""
Tests for the OSSE data-preparation chain that require xarray/pandas (run
inside the project conda env, unlike test_synthetic_obs.py which is
numpy-only):

    conda activate 4dvarnet-daniel
    python tests/test_argo_and_pseudo_obs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap_no_heavy_deps  # noqa: F401

import numpy as np
import pandas as pd
import xarray as xr

import types
sys.modules.setdefault('contrib.argo', _pkg := types.ModuleType('contrib.argo'))
_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'contrib' / 'argo')]

from contrib.argo.qc import sort_pointcloud, reject_pressure_inversions, apply_standard_qc  # noqa: E402
from contrib.argo.virtual import virtualize_profiles  # noqa: E402
from contrib.synthetic_obs.make_pseudo_obs import make_pseudo_obs  # noqa: E402
from contrib.synthetic_obs.build_masks import build_and_serialize_masks  # noqa: E402


def _pointcloud(platforms, cycles, pres, temp=None):
    n = len(pres)
    return xr.Dataset(
        {
            'PLATFORM_NUMBER': ('N_POINTS', np.array(platforms)),
            'CYCLE_NUMBER': ('N_POINTS', np.array(cycles)),
            'PRES': ('N_POINTS', np.array(pres, dtype=float)),
            'TEMP': ('N_POINTS', np.array(temp if temp is not None else np.ones(n), dtype=float)),
        }
    )


def test_inversion_qc_on_unsorted_input_keeps_valid_points():
    """Regression for the missing-sort bug. reject_pressure_inversions only
    compares ADJACENT rows of the same profile (see its docstring: "assumes
    ds is already sorted"). The bug therefore doesn't come from profiles
    being interleaved with each other (cross-profile pairs are already
    screened out by the same_profile check) - it comes from a single
    profile's OWN points not being in pressure order, which argopy does not
    guarantee. Two profiles, each internally out of pressure order but with
    no true inversion once sorted:
        A: 20, 10, 30   (sorted: 10, 20, 30 - monotonic, valid)
        B: 15, 5,  25   (sorted: 5, 15, 25  - monotonic, valid)
    Unsorted, the out-of-order interior points look like inversions to the
    adjacent-pair test and get spuriously dropped.
    """
    ds = _pointcloud(
        platforms=['A', 'A', 'A', 'B', 'B', 'B'],
        cycles=[1, 1, 1, 1, 1, 1],
        pres=[20, 10, 30, 15, 5, 25],
    )
    unsorted_kept = reject_pressure_inversions(ds).sizes['N_POINTS']
    sorted_kept = reject_pressure_inversions(sort_pointcloud(ds)).sizes['N_POINTS']
    assert sorted_kept == 6, sorted_kept          # nothing should be rejected once sorted
    assert unsorted_kept < 6, unsorted_kept        # demonstrates the bug the sort fixes


def test_inversion_qc_still_rejects_true_inversions():
    ds = _pointcloud(['A'] * 4, [1] * 4, [10, 20, 15, 30])  # 15 after 20: inversion
    kept = reject_pressure_inversions(sort_pointcloud(ds))
    # after sorting: 10, 15, 20, 30 -> monotonic, so the sort alone "fixes" order;
    # duplicate-pressure rejection is what remains observable:
    ds2 = _pointcloud(['A'] * 3, [1] * 3, [10, 10, 20])
    kept2 = reject_pressure_inversions(sort_pointcloud(ds2))
    assert kept2.sizes['N_POINTS'] == 2


def _tiny_truth(tmpdir, n_days=6, n_lat=8, n_lon=8, with_depth=False):
    time = pd.date_range('2019-01-01', periods=n_days, freq='D')
    lat = np.linspace(30, 37, n_lat)
    lon = np.linspace(-60, -53, n_lon)
    shape = (n_days, n_lat, n_lon)
    coords = dict(time=time, lat=lat, lon=lon)
    dims = ('time', 'lat', 'lon')
    if with_depth:
        depth = np.array([0.49, 50.0])
        shape = (n_days, len(depth), n_lat, n_lon)
        coords = dict(time=time, depth=depth, lat=lat, lon=lon)
        dims = ('time', 'depth', 'lat', 'lon')
    rng = np.random.default_rng(0)
    values = rng.normal(20, 2, size=shape).astype(np.float32)
    ds = xr.Dataset({'thetao': (dims, values), 'zos': (dims[:1] + dims[-2:], values[:, 0] if with_depth else values)},
                    coords=coords)
    path = str(Path(tmpdir) / ('truth3d.nc' if with_depth else 'truth.nc'))
    ds.to_netcdf(path)
    return path, ds


def test_virtualize_profiles_samples_truth_not_reality(tmp_path='/tmp/nosc_tests'):
    Path(tmp_path).mkdir(exist_ok=True)
    truth_path, truth = _tiny_truth(tmp_path, with_depth=True)
    interp_df = pd.DataFrame(dict(
        profile_id=['A_1', 'A_1', 'B_1'],
        lat=[31.0, 31.0, 36.0], lon=[-59.0, -59.0, -54.0],
        time=pd.to_datetime(['2019-01-02', '2019-01-02', '2019-01-05']),
        depth_level=[0.49, 50.0, 0.49],
        TEMP=[99.0, np.nan, 99.0],   # 99 = fake "real" values; NaN = level not covered
    ))
    out = virtualize_profiles(interp_df, truth_path, value_var='TEMP', truth_var='thetao')
    assert np.isnan(out['TEMP'].iloc[1])                       # coverage preserved
    assert not np.any(np.isclose(out['TEMP'].dropna(), 99.0))  # real values discarded
    expected = truth['thetao'].sel(time='2019-01-02', depth=0.49, lat=31.0, lon=-59.0, method='nearest').item()
    assert np.isclose(out['TEMP'].iloc[0], expected)


def test_pseudo_obs_masking_noise_and_alignment(tmp_path='/tmp/nosc_tests'):
    Path(tmp_path).mkdir(exist_ok=True)
    truth_path, truth = _tiny_truth(tmp_path)
    masks_path = str(Path(tmp_path) / 'masks.pickle')
    build_and_serialize_masks(masks_path, grid_from=truth_path, time_from=truth_path,
                              mission_names=['jason3'])
    out_path = str(Path(tmp_path) / 'pseudo_obs.nc')
    make_pseudo_obs(truth_path, 'zos', masks_path, out_path, noise_std=0.02, seed=7)
    obs = xr.open_dataset(out_path)
    assert obs.sizes['time'] == truth.sizes['time']            # time_from alignment
    observed = obs.obs_mask.values.astype(bool)
    assert np.isnan(obs.zos_obs.values[~observed]).all()       # NaN off-track
    diffs = obs.zos_obs.values[observed] - truth.zos.values[observed]
    assert 0.0 < np.std(diffs) < 0.05                          # noise present, right order
    # determinism: same seed -> identical file
    make_pseudo_obs(truth_path, 'zos', masks_path, str(Path(tmp_path) / 'pseudo_obs2.nc'), noise_std=0.02, seed=7)
    obs2 = xr.open_dataset(str(Path(tmp_path) / 'pseudo_obs2.nc'))
    assert np.array_equal(obs.zos_obs.values, obs2.zos_obs.values, equal_nan=True)


if __name__ == '__main__':
    for fn_name in list(globals()):
        if fn_name.startswith('test_'):
            print(f"{fn_name} ...", end=' ', flush=True)
            globals()[fn_name]()
            print("OK")
    print("All argo/pseudo-obs tests passed.")
