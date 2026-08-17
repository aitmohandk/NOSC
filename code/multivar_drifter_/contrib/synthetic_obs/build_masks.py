"""
Entrypoint reproducing the exact contract of
process_data/mask/glorys_masking.ipynb's get_list_masks/serialize_masks, but
from simulated satellite tracks (contrib.synthetic_obs) instead of a
pre-gridded real nadir product. Output is a pickle of daily 2D masks,
drop-in compatible with the `mask_path` option consumed by
contrib/data_loading/data.py (open_var_dataset, open_glorys12_data).

CLI usage:
    python -m contrib.synthetic_obs.build_masks \
        --grid-from /path/to/reference_grid.nc --n-days 365 \
        --missions jason3 sentinel3a saral --output masks.pickle

Hydra usage (as an entrypoint or a mask_path-producing preprocessing step):
    _target_: contrib.synthetic_obs.build_masks.build_and_serialize_masks
    output_path: ...
    grid_from: ...
    n_days: 365
    mission_names: [jason3, sentinel3a, saral, hy2b]
"""
import argparse
import pickle

import numpy as np
import xarray as xr

from contrib.synthetic_obs.missions import MISSIONS, SIX_SAT_NADIR
from contrib.synthetic_obs.sampling import build_daily_masks


def serialize_masks(path, mask_list):
    with open(path, 'wb') as handle:
        pickle.dump(mask_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def grid_from_dataset(path, lat_name='lat', lon_name='lon'):
    ds = xr.open_dataset(path)
    return np.asarray(ds[lat_name].values, dtype=np.float64), np.asarray(ds[lon_name].values, dtype=np.float64)


def build_and_serialize_masks(
    output_path,
    grid_from=None,
    lat_grid=None,
    lon_grid=None,
    n_days=365,
    mission_names=SIX_SAT_NADIR,
    n_samples_per_orbit=720,
    cross_track_step_km=5.0,
):
    """
    grid_from: path to a NetCDF file to read the target (lat, lon) grid from
        (mirrors gridded_glorys.lat/lon in glorys_masking.ipynb). Ignored if
        lat_grid/lon_grid are given directly.
    """
    if lat_grid is None or lon_grid is None:
        lat_grid, lon_grid = grid_from_dataset(grid_from)

    missions = [MISSIONS[name] for name in mission_names]
    masks = build_daily_masks(
        missions, n_days, np.asarray(lat_grid), np.asarray(lon_grid),
        n_samples_per_orbit=n_samples_per_orbit, cross_track_step_km=cross_track_step_km,
    )
    serialize_masks(output_path, masks)
    return masks


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid-from', required=True, help='NetCDF file to read the target lat/lon grid from')
    parser.add_argument('--n-days', type=int, default=365)
    parser.add_argument('--missions', nargs='+', default=SIX_SAT_NADIR, choices=list(MISSIONS))
    parser.add_argument('--n-samples-per-orbit', type=int, default=720)
    parser.add_argument('--cross-track-step-km', type=float, default=5.0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    build_and_serialize_masks(
        output_path=args.output,
        grid_from=args.grid_from,
        n_days=args.n_days,
        mission_names=args.missions,
        n_samples_per_orbit=args.n_samples_per_orbit,
        cross_track_step_km=args.cross_track_step_km,
    )
    print(f"Wrote {args.n_days} daily masks to {args.output}")


if __name__ == '__main__':
    _cli()
