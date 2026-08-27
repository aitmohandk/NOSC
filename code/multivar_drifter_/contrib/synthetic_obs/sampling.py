"""
Rasterize simulated ground tracks / swaths onto a target lat/lon grid,
producing daily boolean coverage masks with the same (list-of-2D-arrays)
contract as process_data/mask/glorys_masking.ipynb's get_list_masks, so the
output plugs directly into the existing mask_path mechanism
(contrib/data_loading/data.py: open_var_dataset / open_glorys12_data).
"""
import numpy as np

from contrib.synthetic_obs.missions import Mission
from contrib.synthetic_obs.orbits import ground_track, swath_tracks, orbits_for_day

EARTH_CIRCUMFERENCE_KM = 40075.0


def _wrap_lon(lon, lon_min):
    """Wrap longitudes into [lon_min, lon_min + 360)."""
    return (np.asarray(lon) - lon_min) % 360.0 + lon_min


def n_samples_for_grid(lat_grid, lon_grid, oversampling=2.0):
    """
    Along-track samples per orbit needed so consecutive samples fall at most
    one grid cell apart (times `oversampling`), guaranteeing continuous
    rasterized tracks instead of dotted ones.

    A full revolution's ground track is ~one Earth circumference long; with a
    minimum cell size `min_cell_km`, continuous coverage needs at least
    circumference / min_cell_km samples. The previous fixed default (720,
    i.e. ~56 km between samples) left visible gaps on grids finer than ~0.5
    degrees - half the cells along a 1/4-degree track were never marked.
    """
    dlat = np.min(np.abs(np.diff(np.asarray(lat_grid, dtype=float))))
    dlon = np.min(np.abs(np.diff(np.asarray(lon_grid, dtype=float))))
    # meridional cell size; zonal size shrinks with cos(lat), use the grid's
    # max |lat| to bound it.
    km_per_deg = EARTH_CIRCUMFERENCE_KM / 360.0
    min_coslat = np.cos(np.radians(np.max(np.abs(lat_grid))))
    min_cell_km = min(dlat * km_per_deg, dlon * km_per_deg * max(min_coslat, 0.05))
    return int(np.ceil(oversampling * EARTH_CIRCUMFERENCE_KM / min_cell_km))


def nearest_grid_indices(lat_pts, lon_pts, lat_grid, lon_grid):
    """
    Nearest-cell (row, col) indices of points (lat_pts, lon_pts) on a
    monotonically increasing (lat_grid, lon_grid) grid. Points outside the
    grid's latitude range are dropped.
    """
    lon_pts = _wrap_lon(lon_pts, lon_grid.min())

    in_range = (lat_pts >= lat_grid.min()) & (lat_pts <= lat_grid.max())
    lat_pts, lon_pts = lat_pts[in_range], lon_pts[in_range]

    row = np.searchsorted(lat_grid, lat_pts)
    row = np.clip(row, 1, len(lat_grid) - 1)
    row -= (np.abs(lat_grid[row - 1] - lat_pts) <= np.abs(lat_grid[row] - lat_pts))

    col = np.searchsorted(lon_grid, lon_pts) % len(lon_grid)
    col_prev = (col - 1) % len(lon_grid)
    col = np.where(
        np.abs(_wrap_lon(lon_grid[col_prev], lon_grid.min()) - lon_pts)
        <= np.abs(_wrap_lon(lon_grid[col], lon_grid.min()) - lon_pts),
        col_prev, col,
    )

    return row, col


def rasterize_day(missions, day_index, lat_grid, lon_grid, n_samples_per_orbit=None, cross_track_step_km=5.0):
    """
    Boolean (n_lat, n_lon) coverage mask for the union of all given missions
    on one day. n_samples_per_orbit=None (default) auto-scales the along-track
    sampling to the grid resolution (n_samples_for_grid) so tracks rasterize
    without gaps.
    """
    if n_samples_per_orbit is None:
        n_samples_per_orbit = n_samples_for_grid(lat_grid, lon_grid)

    covered = np.zeros((len(lat_grid), len(lon_grid)), dtype=bool)

    for mission in missions:
        for orbit_number in orbits_for_day(mission, day_index):
            lat, lon = ground_track(mission, orbit_number, n_samples=n_samples_per_orbit)
            row, col = nearest_grid_indices(lat, lon, lat_grid, lon_grid)
            covered[row, col] = True

            for swath_lat, swath_lon in swath_tracks(
                mission, orbit_number, n_samples=n_samples_per_orbit, cross_track_step_km=cross_track_step_km
            ):
                row, col = nearest_grid_indices(swath_lat, swath_lon, lat_grid, lon_grid)
                covered[row, col] = True

    return covered


def build_daily_masks(missions, n_days, lat_grid, lon_grid, n_samples_per_orbit=None, cross_track_step_km=5.0):
    """
    List of n_days daily masks (float32 arrays, shape (n_lat, n_lon)), 1.0
    where a simulated pass observed the cell that day and NaN elsewhere -
    matching the pickle contract consumed by contrib/data_loading/data.py's
    mask_input / open_glorys12_data. Day 0 of the list corresponds to
    day_index 0 of the orbit model; the caller is responsible for anchoring
    day 0 to the dataset's first date (see build_masks.build_and_serialize_masks'
    time_from option), since data.py applies masks sequentially by index.
    """
    if n_samples_per_orbit is None:
        n_samples_per_orbit = n_samples_for_grid(lat_grid, lon_grid)

    masks = []
    for day_index in range(n_days):
        covered = rasterize_day(
            missions, day_index, lat_grid, lon_grid,
            n_samples_per_orbit=n_samples_per_orbit, cross_track_step_km=cross_track_step_km,
        )
        mask = np.full(covered.shape, np.nan, dtype=np.float32)
        mask[covered] = 1.0
        masks.append(mask)
    return masks
