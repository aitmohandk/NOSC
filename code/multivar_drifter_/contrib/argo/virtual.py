"""
Virtual ARGO observations for a GLORYS-truth OSSE.

Rationale: in an OSSE whose truth is GLORYS, REAL Argo values are
inconsistent observations (GLORYS assimilates Argo but does not match it
pointwise; training on both teaches the network two contradictory truths for
the same temperature). The consistent construction keeps the REAL sampling
geometry - float positions, dates, and per-profile vertical coverage - but
replaces the measured values by the truth sampled at those points:
"virtual floats" flying through GLORYS.

Pipeline (run_virtual_pipeline):
  1. download real profiles, chunked (download.fetch_argo_profiles_chunked)
  2. standard QC, sorted (qc.apply_standard_qc)
  3. vertical interpolation of the REAL values onto the model depth levels
     (vertical_interp.interp_argo_profiles) - the real values are then
     DISCARDED; the interpolation only serves to determine, per profile,
     which depth levels the float actually covered (NaN outside its range),
     so a virtual float never reports a level the real one did not measure.
  4. virtualize_profiles: vectorized nearest-neighbour sampling of the truth
     at each (time, depth, lat, lon)
  5. gridding via build_argo_dataset.build_and_save_argo_datasets, unchanged
     (one NetCDF per (var, depth)), ready for `multivar:` entries - typically
     input_arch: prior_input (observations the network learns to USE), see
     config/vars/argo_virtual_thetao_depths.yaml.
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from contrib.argo.build_argo_dataset import build_and_save_argo_datasets
from contrib.argo.download import fetch_argo_profiles_chunked
from contrib.argo.qc import apply_standard_qc
from contrib.argo.vertical_interp import interp_argo_profiles


def virtualize_profiles(interp_df, truth_path, value_var='TEMP', truth_var='thetao',
                        time_tolerance='2D'):
    """
    Replace `value_var` in interp_df (output of interp_argo_profiles) by the
    truth dataset sampled at each row's (time, depth_level, lat, lon), using
    vectorized nearest-neighbour selection. Rows where the real profile had
    no valid value at that level (NaN) stay NaN, preserving per-profile
    vertical coverage. Rows falling outside the truth's time axis by more
    than `time_tolerance` are set to NaN (guards against silent edge
    matching).
    """
    da = xr.open_dataset(truth_path)[truth_var]
    if 'latitude' in da.dims:
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

    df = interp_df.copy()
    covered = np.isfinite(df[value_var].values)

    pts = df.loc[covered]
    indexers = dict(
        time=xr.DataArray(pd.to_datetime(pts['time']).values, dims='pts'),
        lat=xr.DataArray(pts['lat'].values, dims='pts'),
        lon=xr.DataArray(pts['lon'].values, dims='pts'),
    )
    if 'depth' in da.dims:
        indexers['depth'] = xr.DataArray(pts['depth_level'].values, dims='pts')

    sampled = da.sel(**indexers, method='nearest')

    # guard: nearest-time match must be within tolerance of the profile date
    dt = np.abs(sampled['time'].values - pd.to_datetime(pts['time']).values)
    ok = dt <= pd.Timedelta(time_tolerance)
    values = np.where(ok, sampled.values, np.nan)

    out = np.full(len(df), np.nan)
    out[np.flatnonzero(covered)] = values
    df[value_var] = out
    return df


def resolve_depth_indices(truth_path, depth_indices):
    """Depth values (m) at the given positions of the truth file's depth axis
    (same convention as the depth_index key of multivar entries)."""
    depth = xr.open_dataset(truth_path)['depth'].values
    bad = [i for i in depth_indices if i >= len(depth) or i < 0]
    if bad:
        raise ValueError(f"depth indices {bad} out of range (file has {len(depth)} levels)")
    return [float(depth[int(i)]) for i in depth_indices]


def run_virtual_pipeline(
    lon_min, lon_max, lat_min, lat_max, start_date, end_date, depths,
    lat_grid, lon_grid, output_dir, truth_path, truth_var='thetao',
    value_var='TEMP', out_var_name='thetao_vargo', mode='standard',
    spike_thresholds=None, fetch_freq='MS', depth_indices=None,
):
    """depths: depth values in meters, OR None with depth_indices set (positions
    in the truth file's depth axis, resolved here - matches the depth_index
    convention of config/depths/*.yaml). With depth_indices, output files are
    named {out_var_name}_d{index:02d}.nc to match the generated
    config/vars/argo_virtual_*_{suffix}.yaml fragments."""
    if depth_indices is not None:
        depths = resolve_depth_indices(truth_path, depth_indices)
        print(f"[virtual argo] depth indices {list(depth_indices)} -> values (m) "
              f"{[round(d, 2) for d in depths]}")
    print(f"[virtual argo] fetching real profile geometry {start_date}..{end_date} (chunked)")
    raw = fetch_argo_profiles_chunked(lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                                      freq=fetch_freq, min_depth=0, max_depth=max(depths) + 50, mode=mode)

    print("[virtual argo] QC (sorted)")
    qcd = apply_standard_qc(raw, value_vars=(value_var,), spike_thresholds=spike_thresholds)

    print(f"[virtual argo] vertical coverage on {len(depths)} depth levels")
    interp_df = interp_argo_profiles(qcd, depths, value_vars=(value_var,))

    print(f"[virtual argo] sampling truth {truth_var} from {truth_path}")
    virtual_df = virtualize_profiles(interp_df, truth_path, value_var=value_var, truth_var=truth_var)

    print(f"[virtual argo] gridding -> {output_dir}")
    paths = build_and_save_argo_datasets(
        virtual_df, (value_var,), depths, lat_grid, lon_grid, start_date, end_date,
        output_dir, var_name_map={value_var: out_var_name},
    )
    if depth_indices is not None:
        from pathlib import Path as _P
        renamed = {}
        for (vv, dval), path in paths.items():
            idx = depth_indices[depths.index(dval)]
            new = _P(path).with_name(f"{out_var_name}_d{int(idx):02d}.nc")
            _P(path).rename(new)
            renamed[(vv, int(idx))] = str(new)
        paths = renamed
    for key, path in paths.items():
        print(f"  {key} -> {path}")
    return paths


def _cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid-from', required=True)
    p.add_argument('--truth-path', required=True, help='GLORYS multi-depth NetCDF (the OSSE truth)')
    p.add_argument('--truth-var', default='thetao')
    p.add_argument('--start-date', required=True)
    p.add_argument('--end-date', required=True)
    p.add_argument('--lon-min', type=float, default=-180)
    p.add_argument('--lon-max', type=float, default=180)
    p.add_argument('--lat-min', type=float, default=-70)
    p.add_argument('--lat-max', type=float, default=70)
    p.add_argument('--depths', type=float, nargs='+', default=None, help='depth values in meters')
    p.add_argument('--depth-indices', type=int, nargs='+', default=None,
                   help='positions in the truth depth axis (preferred; matches config/depths/*.yaml)')
    p.add_argument('--out-var-name', default='thetao_vargo')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    ref = xr.open_dataset(args.grid_from)
    if 'latitude' in ref.dims:
        ref = ref.rename({'latitude': 'lat', 'longitude': 'lon'})
    lat_grid, lon_grid = np.asarray(ref.lat.values), np.asarray(ref.lon.values)
    run_virtual_pipeline(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        args.start_date, args.end_date, args.depths, lat_grid, lon_grid,
        args.output_dir, args.truth_path, truth_var=args.truth_var,
        out_var_name=args.out_var_name, depth_indices=args.depth_indices,
    )
    if args.depths is None and args.depth_indices is None:
        raise SystemExit('provide --depths or --depth-indices')


if __name__ == '__main__':
    _cli()
