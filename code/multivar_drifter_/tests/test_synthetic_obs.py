"""
Physical sanity tests for the synthetic observation generator (pure numpy,
no torch/xarray needed): orbital plausibility, daily coverage statistics,
along-track continuity, and repeat-cycle closure.

Run: python -m pytest tests/test_synthetic_obs.py  (or python tests/test_synthetic_obs.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap_no_heavy_deps  # noqa: F401  (registers contrib pkg without heavy deps)

from contrib.synthetic_obs.missions import MISSIONS, SIX_SAT_NADIR, validate_missions, Mission
from contrib.synthetic_obs.sampling import rasterize_day, n_samples_for_grid
from contrib.synthetic_obs.orbits import ground_track


def quarter_degree_grid():
    lat = np.arange(-69.875, 70, 0.25)
    lon = np.arange(-179.875, 180, 0.25)
    return lat, lon


def test_orbital_periods_plausible():
    for m in MISSIONS.values():
        assert 90 * 60 <= m.orbital_period_s <= 130 * 60, (m.name, m.orbital_period_s / 60)
        assert 11.0 <= m.orbits_per_day <= 15.5, (m.name, m.orbits_per_day)


def test_validation_catches_passes_vs_orbits_confusion():
    bad = Mission('bad_jason', 66.04, 9.9156, 254)  # passes mistaken for orbits
    try:
        validate_missions([bad])
    except ValueError:
        return
    raise AssertionError("validate_missions accepted a 56-minute orbit")


def test_daily_nadir_coverage_fraction():
    """A single nadir altimeter covers ~1.5-3.5% of a 1/4-degree +-70 grid per
    day (track length/day ~ orbits_per_day x 40,075 km over ~806k cells of
    ~28 km). The pre-fix Jason-3 (doubled orbit count) gave ~2x this."""
    lat, lon = quarter_degree_grid()
    n_cells = len(lat) * len(lon)
    for name in ["jason3", "sentinel3a", "saral", "hy2b"]:
        cov = rasterize_day([MISSIONS[name]], day_index=3, lat_grid=lat, lon_grid=lon)
        frac = cov.sum() / n_cells
        assert 0.015 <= frac <= 0.045, (name, frac)


def test_coverage_regression_vs_passes_bug():
    """The pre-fix Jason-3 (254 'orbits' = passes) roughly doubles daily
    coverage; ensure the corrected mission stays well below that regime."""
    lat, lon = quarter_degree_grid()
    n_cells = len(lat) * len(lon)
    good = rasterize_day([MISSIONS['jason3']], day_index=3, lat_grid=lat, lon_grid=lon).sum() / n_cells
    buggy_mission = Mission('jason3_buggy', 66.04, 9.9156, 254)
    buggy = rasterize_day([buggy_mission], day_index=3, lat_grid=lat, lon_grid=lon).sum() / n_cells
    assert buggy > 1.6 * good, (good, buggy)
    assert good < 0.045 < buggy, (good, buggy)


def test_track_continuity_no_dotted_tracks():
    """With auto along-track sampling, consecutive samples of one orbit land
    on identical or 8-adjacent grid cells (no gaps) for > 99% of steps."""
    lat_grid, lon_grid = quarter_degree_grid()
    n = n_samples_for_grid(lat_grid, lon_grid)
    assert n >= 2500, n  # 1/4 deg needs ~2900, old default 720 was far too low
    m = MISSIONS['jason3']
    lat, lon = ground_track(m, orbit_number=5, n_samples=n)
    from contrib.synthetic_obs.sampling import nearest_grid_indices
    row, col = nearest_grid_indices(np.asarray(lat), np.asarray(lon), lat_grid, lon_grid)
    drow = np.abs(np.diff(row))
    dcol = np.abs(np.diff(col))
    dcol = np.minimum(dcol, len(lon_grid) - dcol)  # periodic longitude
    adjacent = (drow <= 1) & (dcol <= 1)
    assert adjacent.mean() > 0.99, adjacent.mean()


def test_repeat_cycle_closure():
    """After exactly one repeat cycle, the ground track must (nearly) repeat:
    that is the defining property the closed-form model is built on."""
    m = MISSIONS['jason3']
    lat0, lon0 = ground_track(m, orbit_number=0, n_samples=360)
    latN, lonN = ground_track(m, orbit_number=m.orbits_per_cycle, n_samples=360)
    dlon = (np.asarray(lonN) - np.asarray(lon0) + 180) % 360 - 180
    assert np.abs(np.asarray(latN) - np.asarray(lat0)).max() < 1e-6
    assert np.abs(dlon).max() < 0.5  # sub-cell closure


if __name__ == '__main__':
    for fn_name in list(globals()):
        if fn_name.startswith('test_'):
            print(f"{fn_name} ...", end=' ', flush=True)
            globals()[fn_name]()
            print("OK")
    print("All synthetic_obs tests passed.")
