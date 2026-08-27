"""
Entrypoint reproducing the exact contract of
process_data/mask/glorys_masking.ipynb's get_list_masks/serialize_masks, but
from simulated satellite tracks (contrib.synthetic_obs) instead of a
pre-gridded real nadir product. Output is a pickle of daily 2D masks,
drop-in compatible with the `mask_path` option consumed by
contrib/data_loading/data.py 
(which applies masks SEQUENTIALLY BY INDEX: mask day 0 -> dataset time step 0).

To guarantee that alignment, prefer `time_from=<reference .nc>`: n_days and
day 0 are then taken from the reference dataset's own time axis instead of a
manually supplied n_days.

Optionally also writes the same coverage as a NetCDF (`output_netcdf`) with
variables `obs_mask` (1.0 observed / 0.0 not) on the reference time axis -
usable directly as an explicit mask input channel via a standard multivar
entry (input_arch: prior_input), no core-code change needed.

CLI usage:
    python -m contrib.synthetic_obs.build_masks \
        --grid-from /path/to/reference_grid.nc --time-from /path/to/reference_grid.nc \
        --missions jason3 sentinel3a saral --output masks.pickle \
        [--output-netcdf masks.nc] [--n-days 365]

Hydra usage (as an entrypoint or a mask_path-producing preprocessing step):
    _target_: contrib.synthetic_obs.build_masks.build_and_serialize_masks
    output_path: ...
    grid_from: ...
    time_from: ...          # preferred over n_days
    output_netcdf: ...      # optional explicit-mask-channel NetCDF
    mission_names: [jason3, sentinel3a, saral, hy2b]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from contrib.synthetic_obs.missions import MISSIONS, SIX_SAT_NADIR, validate_missions
from contrib.synthetic_obs.sampling import build_daily_masks


def serialize_masks(path, mask_list):
    with open(path, 'wb') as handle:
        pickle.dump(mask_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def grid_from_dataset(path, lat_name='lat', lon_name='lon'):
    ds = xr.open_dataset(path)
    if lat_name not in ds.variables and 'latitude' in ds.variables:
        lat_name, lon_name = 'latitude', 'longitude'
    return np.asarray(ds[lat_name].values, dtype=np.float64), np.asarray(ds[lon_name].values, dtype=np.float64)


def time_axis_from_dataset(path, time_name='time'):
    """Daily-normalized time axis of the reference dataset (defines n_days AND day-0 anchoring)."""
    times = pd.DatetimeIndex(xr.open_dataset(path)[time_name].values).normalize()
    return times


def write_masks_netcdf(path, mask_list, lat_grid, lon_grid, times):
    """Coverage as a NetCDF: obs_mask = 1.0 observed / 0.0 not, on the reference time axis."""
    arr = np.nan_to_num(np.stack(mask_list, axis=0), nan=0.0).astype(np.float32)
    ds = xr.Dataset(
        {'obs_mask': (('time', 'lat', 'lon'), arr)},
        coords=dict(time=np.asarray(times), lat=np.asarray(lat_grid), lon=np.asarray(lon_grid)),
    )
    ds.to_netcdf(path)
    return path


def build_and_serialize_masks(
    output_path,
    grid_from=None,
    lat_grid=None,
    lon_grid=None,
    n_days=None,
    time_from=None,
    output_netcdf=None,
    mission_names=SIX_SAT_NADIR,
    n_samples_per_orbit=None,
    cross_track_step_km=5.0,
    skip_if_exists=True,
):
    """
    grid_from: path to a NetCDF file to read the target (lat, lon) grid from.
        Ignored if lat_grid/lon_grid are given directly.
    time_from: path to a NetCDF file whose `time` axis defines BOTH n_days and
        the day-0 anchoring of the mask sequence (recommended: pass the same
        file the masked variable is loaded from, so alignment holds by
        construction). Mutually consistent with n_days: if both are given,
        n_days must not exceed the reference axis length.
    """
    if skip_if_exists and Path(output_path).exists() and (
            output_netcdf is None or Path(output_netcdf).exists()):
        print(f"[build_masks] {output_path} exists - skipping regeneration "
              f"(delete the file or pass skip_if_exists=False to force; "
              f"generation is seeded/deterministic, so a regenerated file is bit-identical)")
        with open(output_path, 'rb') as f:
            return pickle.load(f)

    if lat_grid is None or lon_grid is None:
        lat_grid, lon_grid = grid_from_dataset(grid_from)

    times = None
    if time_from is not None:
        times = time_axis_from_dataset(time_from)
        if n_days is None:
            n_days = len(times)
        elif n_days > len(times):
            raise ValueError(f"n_days={n_days} exceeds the reference time axis length {len(times)}")
        times = times[:n_days]
    elif n_days is None:
        raise ValueError("either n_days or time_from must be provided")

    missions = [MISSIONS[name] for name in mission_names]
    validate_missions(missions)
    masks = build_daily_masks(
        missions, n_days, np.asarray(lat_grid), np.asarray(lon_grid),
        n_samples_per_orbit=n_samples_per_orbit, cross_track_step_km=cross_track_step_km,
    )
    serialize_masks(output_path, masks)

    if output_netcdf is not None:
        if times is None:
            times = pd.date_range('2000-01-01', periods=n_days, freq='D')  # placeholder axis
        write_masks_netcdf(output_netcdf, masks, lat_grid, lon_grid, times)

    return masks


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid-from', required=True, help='NetCDF file to read the target lat/lon grid from')
    parser.add_argument('--time-from', default=None, help='NetCDF file whose time axis anchors and sizes the mask sequence (recommended)')
    parser.add_argument('--n-days', type=int, default=None)
    parser.add_argument('--missions', nargs='+', default=SIX_SAT_NADIR, choices=list(MISSIONS))
    parser.add_argument('--n-samples-per-orbit', type=int, default=None, help='default: auto from grid resolution')
    parser.add_argument('--cross-track-step-km', type=float, default=5.0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--output-netcdf', default=None)
    args = parser.parse_args()

    masks = build_and_serialize_masks(
        output_path=args.output,
        grid_from=args.grid_from,
        time_from=args.time_from,
        n_days=args.n_days,
        output_netcdf=args.output_netcdf,
        mission_names=args.missions,
        n_samples_per_orbit=args.n_samples_per_orbit,
        cross_track_step_km=args.cross_track_step_km,
    )
    print(f"Wrote {len(masks)} daily masks to {args.output}")


if __name__ == '__main__':
    _cli()
