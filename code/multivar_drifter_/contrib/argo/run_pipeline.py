"""
End-to-end CLI driver chaining contrib/argo's building blocks (download -> QC
-> vertical interpolation -> gridding), producing the per-(variable,depth)
NetCDF files referenced by config/vars/thetao_argo_depths.yaml.

CLI usage:
    python -m contrib.argo.run_pipeline \
        --grid-from /path/to/reference_grid.nc \
        --start-date 2010-01-01 --end-date 2020-01-01 \
        --lon-min -180 --lon-max 180 --lat-min -70 --lat-max 70 \
        --depths 0.49 15 50 100 200 \
        --output-dir /path/to/argo/gridded
"""
import argparse

import numpy as np
import xarray as xr

from contrib.argo.build_argo_dataset import build_and_save_argo_datasets
from contrib.argo.download import fetch_argo_profiles
from contrib.argo.qc import apply_standard_qc
from contrib.argo.vertical_interp import interp_argo_profiles


def run_pipeline(
    lon_min, lon_max, lat_min, lat_max, start_date, end_date, depths,
    lat_grid, lon_grid, output_dir, value_vars=('TEMP',), var_name_map=None,
    mode='standard', spike_thresholds=None,
):
    print(f"Fetching Argo profiles [{lon_min},{lon_max}]x[{lat_min},{lat_max}] "
          f"{start_date}..{end_date} (mode={mode})")
    raw = fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                              min_depth=0, max_depth=max(depths) + 50, mode=mode)

    print("Applying QC")
    qcd = apply_standard_qc(raw, value_vars=value_vars, spike_thresholds=spike_thresholds)
    print(f"  kept {qcd.sizes.get('N_POINTS', qcd.sizes.get('N_PROF'))} / "
          f"{raw.sizes.get('N_POINTS', raw.sizes.get('N_PROF'))} points")

    print(f"Interpolating onto {len(depths)} depth levels")
    interp_df = interp_argo_profiles(qcd, depths, value_vars=value_vars)

    print(f"Gridding onto target grid and writing to {output_dir}")
    paths = build_and_save_argo_datasets(
        interp_df, value_vars, depths, lat_grid, lon_grid, start_date, end_date,
        output_dir, var_name_map=var_name_map,
    )
    for key, path in paths.items():
        print(f"  {key} -> {path}")
    return paths


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid-from', required=True, help='NetCDF file to read the target lat/lon grid from')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--lon-min', type=float, default=-180)
    parser.add_argument('--lon-max', type=float, default=180)
    parser.add_argument('--lat-min', type=float, default=-70)
    parser.add_argument('--lat-max', type=float, default=70)
    parser.add_argument('--depths', type=float, nargs='+', required=True)
    parser.add_argument('--value-vars', nargs='+', default=['TEMP'])
    parser.add_argument('--mode', default='standard', choices=['standard', 'expert'])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    ref = xr.open_dataset(args.grid_from)
    lat_grid, lon_grid = np.asarray(ref.lat.values), np.asarray(ref.lon.values)

    run_pipeline(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        args.start_date, args.end_date, args.depths, lat_grid, lon_grid,
        args.output_dir, value_vars=tuple(args.value_vars), mode=args.mode,
    )


if __name__ == '__main__':
    _cli()
