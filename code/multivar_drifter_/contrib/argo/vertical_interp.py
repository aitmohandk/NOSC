"""
Interpolate QC'd Argo profiles (irregular pressure levels) onto the fixed
depth-level list used by the multi-depth Glorys extraction (Phase 2,
contrib/data_loading/data.py's `depth_level` / config/vars/*_depths.yaml),
so Argo values at e.g. 15m/50m/100m directly compare against the model's
thetao_15m/thetao_50m/thetao_100m outputs.
"""
import numpy as np
import pandas as pd
import xarray as xr


def interp_profile_to_levels(pres, values, target_depths):
    """1D linear interpolation of one profile onto target_depths; NaN outside the profile's pressure range."""
    pres = np.asarray(pres, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(pres) & np.isfinite(values)
    pres, values = pres[valid], values[valid]

    if pres.size < 2:
        return np.full(len(target_depths), np.nan)

    order = np.argsort(pres)
    pres, values = pres[order], values[order]
    pres, unique_idx = np.unique(pres, return_index=True)
    values = values[unique_idx]

    return np.interp(target_depths, pres, values, left=np.nan, right=np.nan)


def interp_argo_profiles(
    ds: xr.Dataset,
    target_depths,
    value_vars=('TEMP', 'PSAL'),
    pres_var='PRES',
    profile_id_var='PLATFORM_NUMBER',
    cycle_var='CYCLE_NUMBER',
    lat_var='LATITUDE',
    lon_var='LONGITUDE',
    time_var='TIME',
):
    """
    Group point-cloud Argo observations (N_POINTS layout) by profile and
    interpolate each one onto `target_depths`.

    Returns a pandas.DataFrame with one row per (profile, depth_level):
    columns [profile_id, lat, lon, time, depth_level, <value_vars...>].
    A DataFrame (not an xr.Dataset) is used because profiles are irregular
    point observations, not a regular grid - this is the "point-cloud" mode
    referenced in the architecture plan, meant for offline validation.
    """
    columns = [pres_var, *value_vars, lat_var, lon_var, time_var, profile_id_var, cycle_var]
    df = ds[columns].to_dataframe().reset_index(drop=True)

    rows = []
    for (platform, cycle), group in df.groupby([profile_id_var, cycle_var]):
        depths_out = {'depth_level': target_depths}
        for value_var in value_vars:
            depths_out[value_var] = interp_profile_to_levels(group[pres_var], group[value_var], target_depths)

        n = len(target_depths)
        rows.append(pd.DataFrame({
            'profile_id': [f'{platform}_{cycle}'] * n,
            'lat': [group[lat_var].iloc[0]] * n,
            'lon': [group[lon_var].iloc[0]] * n,
            'time': [group[time_var].iloc[0]] * n,
            **depths_out,
        }))

    if not rows:
        return pd.DataFrame(columns=['profile_id', 'lat', 'lon', 'time', 'depth_level', *value_vars])

    return pd.concat(rows, ignore_index=True)
