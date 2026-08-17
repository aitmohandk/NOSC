"""
Point-cloud colocation of QC'd, depth-interpolated Argo profiles
(vertical_interp.interp_argo_profiles) against model reconstructions, for
offline validation - independent of the training loop, analogous to how
contrib/ose_pipeline/ose_metrics_pipeline.py evaluates reconstructions
against real along-track altimetry data.

This is Phase 3 (validation-only) of the ARGO integration. Promoting Argo to
a training target (gridded mode, Phase 4) is a separate, not-yet-implemented
step (see the architecture plan).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _as_dataarray(source, var='out'):
    if isinstance(source, (str, Path)):
        return xr.open_dataset(source)[var]
    return source


def colocate_profiles_pointwise(argo_df, reconstruction_by_depth, value_col='TEMP', reconstruction_var='out'):
    """
    argo_df: output of vertical_interp.interp_argo_profiles (columns
        profile_id, lat, lon, time, depth_level, <value_col>...).
    reconstruction_by_depth: {depth_level: path_or_DataArray}, one
        reconstruction per depth level (e.g. the test_data_dim{i}.nc files
        written by Multivar4dVarNet.on_test_epoch_end for the thetao_*m
        output dims of a Phase-2 xp run - the caller must map output_dim
        index -> depth_level using their own multivar dict ordering).

    Returns a DataFrame with one row per successfully colocated
    (profile, depth) pair: [profile_id, depth_level, lat, lon, time,
    argo_value, model_value].
    """
    reconstructions = {depth: _as_dataarray(src, reconstruction_var) for depth, src in reconstruction_by_depth.items()}
    available_depths = np.array(sorted(reconstructions))

    rows = []
    for depth_level, group in argo_df.dropna(subset=[value_col]).groupby('depth_level'):
        nearest_depth = available_depths[np.argmin(np.abs(available_depths - depth_level))]
        da = reconstructions[nearest_depth]

        for _, row in group.iterrows():
            try:
                model_value = da.sel(lat=row['lat'], lon=row['lon'], time=row['time'], method='nearest').item()
            except (KeyError, IndexError):
                continue
            if not np.isfinite(model_value):
                continue
            rows.append(dict(
                profile_id=row['profile_id'], depth_level=depth_level,
                lat=row['lat'], lon=row['lon'], time=row['time'],
                argo_value=row[value_col], model_value=model_value,
            ))

    return pd.DataFrame(rows, columns=['profile_id', 'depth_level', 'lat', 'lon', 'time', 'argo_value', 'model_value'])


def compute_validation_metrics(colocated_df):
    """Per-depth-level RMSE/bias/std/count of (model - argo), independent of the training loop."""
    df = colocated_df.assign(error=colocated_df['model_value'] - colocated_df['argo_value'])
    return df.groupby('depth_level').agg(
        rmse=('error', lambda e: float(np.sqrt(np.mean(e ** 2)))),
        bias=('error', 'mean'),
        std=('error', 'std'),
        n=('error', 'count'),
    )
