"""
Phase 4 (gridded / training-target mode) of the ARGO integration: bins QC'd,
depth-interpolated Argo profiles (vertical_interp.interp_argo_profiles) onto
a reference (time, lat, lon) grid, one NetCDF per (value_var, depth_level) -
the exact same "sparse point obs -> daily-gridded NetCDF -> multivar
full_output entry" pattern already used for surface drifters in
process_data/drifters/make_daily_uv_map_aoml.py (there: scipy
binned_statistic_2d, daily mean, u_drifter/v_drifter). Output files plug
straight into a `multivar:` entry with input_arch: no_input, output_arch:
full_output, exactly like u_drifter/v_drifter do today.

Promote from Phase 3 (validation-only) to this module only once the
depth-channel training (Phase 2) and offline validation (Phase 3, colocate.py)
both look correct - see the architecture plan's phase ordering rationale.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def _edges_from_centers(centers):
    """Bin edges implied by a 1D array of grid-cell centers (possibly non-uniform spacing)."""
    centers = np.asarray(centers, dtype=float)
    mid = (centers[:-1] + centers[1:]) / 2.0
    first_edge = centers[0] - (mid[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first_edge], mid, [last_edge]])


def bin_daily_grid(day_df, value_var, lat_grid, lon_grid):
    """One day's (lon, lat, value) points -> (lat, lon) grid of daily means, NaN where no obs."""
    if day_df.empty:
        return np.full((len(lat_grid), len(lon_grid)), np.nan, dtype=np.float32)

    lon_edges = _edges_from_centers(lon_grid)
    lat_edges = _edges_from_centers(lat_grid)
    grid, _, _, _ = stats.binned_statistic_2d(
        day_df['lon'].values, day_df['lat'].values, day_df[value_var].values,
        statistic='mean', bins=[lon_edges, lat_edges],
    )
    return grid.T.astype(np.float32)  # binned_statistic_2d returns (lon, lat) -> transpose to (lat, lon)


def build_gridded_argo_dataset(interp_df, value_var, depth_level, lat_grid, lon_grid, start_date, end_date, var_name=None):
    """
    interp_df: output of vertical_interp.interp_argo_profiles (already QC'd
        upstream via qc.apply_standard_qc).
    Returns an xr.Dataset with one variable `var_name` (default
        f"{value_var.lower()}_argo"), dims (time, lat, lon).
    """
    var_name = var_name or f'{value_var.lower()}_argo'
    lat_grid, lon_grid = np.asarray(lat_grid), np.asarray(lon_grid)

    depth_df = interp_df[np.isclose(interp_df['depth_level'], depth_level)].dropna(subset=[value_var])
    depth_df = depth_df.assign(day=pd.to_datetime(depth_df['time']).dt.normalize())

    dates = pd.date_range(start_date, end_date, freq='D')
    daily_arrays = []
    for day in dates:
        day_df = depth_df[depth_df['day'] == day]
        grid = bin_daily_grid(day_df, value_var, lat_grid, lon_grid)
        daily_arrays.append(
            xr.DataArray(grid, dims=('lat', 'lon'), coords=dict(lat=lat_grid, lon=lon_grid))
            .expand_dims(time=[day])
        )

    da = xr.concat(daily_arrays, dim='time').astype(np.float32)
    return xr.Dataset({var_name: da})


def build_and_save_argo_datasets(
    interp_df, value_vars, depth_levels, lat_grid, lon_grid, start_date, end_date,
    output_dir, var_name_map=None,
):
    """
    Build and save one NetCDF per (value_var, depth_level), named
    f"{output_dir}/{var_name}_{depth}m.nc" with a matching internal variable
    name - ready to be referenced as `var_path`/`var_name` in a multivar:
    entry (input_arch: no_input, output_arch: full_output), mirroring the
    u_drifter/v_drifter entries in existing xp configs.

    var_name_map: optional {value_var: output_var_name} (default
        f"{value_var.lower()}_argo", e.g. TEMP -> "temp_argo"). Required to
        be distinct across value_vars, since it also names the output files.

    Returns {(value_var, depth_level): output_path}.
    """
    var_name_map = var_name_map or {v: f'{v.lower()}_argo' for v in value_vars}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for value_var in value_vars:
        var_name = var_name_map[value_var]
        for depth_level in depth_levels:
            ds = build_gridded_argo_dataset(
                interp_df, value_var, depth_level, lat_grid, lon_grid, start_date, end_date, var_name=var_name,
            )
            depth_str = str(int(depth_level)) if float(depth_level).is_integer() else str(depth_level)
            path = output_dir / f'{var_name}_{depth_str}m.nc'
            ds.to_netcdf(path)
            output_paths[(value_var, depth_level)] = path

    return output_paths
