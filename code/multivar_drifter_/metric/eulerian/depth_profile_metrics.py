"""
Per-depth evaluation of a multivariate OSSE reconstruction: nRMSE, bias and
effective resolution for every output variable, read from the
test_data_dim{i}.nc files (variable 'out') written at test time, compared to
the TRUTH files directly (the test files only store the reconstruction).

The skill-vs-depth curves this produces are the central result of the 3D
extension - an aggregate score over ~64 heterogeneous channels is
uninterpretable.

Effective resolution: shortest wavelength at which the isotropic PSD of the
error stays below half the isotropic PSD of the truth (the standard
SSH-mapping definition), computed per time step then median-aggregated.

Usage:
    python -m metric.eulerian.depth_profile_metrics \
        --test-dir <hydra_run_dir/logs> \
        --truth-multidepth <glorys_multidepth.nc> --truth-surface <glorys_surface.nc> \
        --time-start 2019-01-01 --time-end 2019-12-31 \
        --output-csv depth_metrics.csv

Variable mapping: output_var_names.json (written by the model at test start)
gives the ordered names; names follow the generated-fragment convention
{var}_d{index} (depth_index) or {var}_{depth}m (depth_level), zos_tgt for the
surface target. Use --manifest to override for custom naming.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FRAG_IDX = re.compile(r'^(?P<var>[a-z]+)_d(?P<idx>\d+)$')
FRAG_VAL = re.compile(r'^(?P<var>[a-z]+)_(?P<val>[\d.]+)m$')


def parse_var_name(name):
    """-> (truth_var, depth_index|None, depth_value|None) or (surface var, None, None)."""
    m = FRAG_IDX.match(name)
    if m and not name.startswith('argo_'):
        return m.group('var'), int(m.group('idx')), None
    m = FRAG_VAL.match(name)
    if m:
        return m.group('var'), None, float(m.group('val'))
    if name in ('zos_tgt', 'ssh_tgt'):
        return 'zos', None, None
    if name in ('sst_tgt',):
        return 'thetao', None, None
    return name, None, None


def isotropic_psd(field2d):
    """Radially averaged power spectral density of a 2D field (NaN -> 0-mean fill)."""
    f = np.nan_to_num(field2d - np.nanmean(field2d))
    ny, nx = f.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    k_bins = np.linspace(0, 0.5, min(ny, nx) // 2)
    which = np.digitize(kr.ravel(), k_bins)
    psd = np.array([spec.ravel()[which == b].mean() if np.any(which == b) else np.nan
                    for b in range(1, len(k_bins))])
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    return k_centers, psd


def effective_resolution_km(err2d, tgt2d, dx_km):
    """Wavelength (km) where PSD_err/PSD_tgt first exceeds 0.5, scanning from
    large to small scales; NaN if never resolved."""
    k, psd_err = isotropic_psd(err2d)
    _, psd_tgt = isotropic_psd(tgt2d)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = psd_err / psd_tgt
    valid = np.isfinite(ratio) & (k > 0)
    k, ratio = k[valid], ratio[valid]
    above = np.flatnonzero(ratio >= 0.5)
    if len(above) == 0:
        return np.nan  # resolved at all scales down to the grid
    return dx_km / k[above[0]]  # wavelength = dx / (cycles per px)


def evaluate_dim(rec_path, truth_da, time_slice, dx_km, stride=5):
    rec = xr.open_dataset(rec_path)['out'].sel(time=time_slice)
    tgt = truth_da.sel(time=rec['time'], lat=rec['lat'], lon=rec['lon'], method='nearest')
    tgt = tgt.assign_coords(time=rec['time'], lat=rec['lat'], lon=rec['lon'])
    err = (rec - tgt)
    finite = np.isfinite(err.values) & np.isfinite(tgt.values)
    e, t = err.values[finite], tgt.values[finite]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    std_t = float(np.std(t))
    res = [effective_resolution_km(err.isel(time=i).values, tgt.isel(time=i).values, dx_km)
           for i in range(0, rec.sizes['time'], stride)]
    return dict(rmse=rmse, nrmse=rmse / std_t if std_t > 0 else np.nan,
                bias=float(np.mean(e)), tgt_std=std_t,
                eff_res_km=float(np.nanmedian(res)) if len(res) else np.nan)


def main(test_dir, truth_multidepth, truth_surface, time_start, time_end,
         output_csv, manifest=None, dx_km=8.0):
    test_dir = Path(test_dir)
    names = json.load(open(manifest or test_dir / 'output_var_names.json'))
    truths = {}
    md = xr.open_dataset(truth_multidepth)
    sf = xr.open_dataset(truth_surface)
    if 'latitude' in md.dims:
        md = md.rename({'latitude': 'lat', 'longitude': 'lon'})
    if 'latitude' in sf.dims:
        sf = sf.rename({'latitude': 'lat', 'longitude': 'lon'})
    time_slice = slice(time_start, time_end)

    rows = []
    for dim, name in enumerate(names):
        rec_path = test_dir / f'test_data_dim{dim}.nc'
        if not rec_path.exists():
            print(f"missing {rec_path}, skipping {name}")
            continue
        var, idx, val = parse_var_name(name)
        if var in md and 'depth' in md[var].dims and (idx is not None or val is not None):
            da = md[var].isel(depth=idx) if idx is not None else md[var].sel(depth=val, method='nearest')
            depth_m = float(da['depth'].values)
        elif var in sf:
            da, depth_m = sf[var], 0.0
        elif var in md:
            da, depth_m = md[var], 0.0
        else:
            print(f"no truth for '{name}' (parsed var '{var}'), skipping"); continue
        metrics = evaluate_dim(rec_path, da, time_slice, dx_km)
        rows.append(dict(dim=dim, name=name, var=var, depth_m=depth_m, **metrics))
        print(f"[{dim:02d}] {name:16s} depth={depth_m:7.2f}m  nRMSE={metrics['nrmse']:.3f}  "
              f"bias={metrics['bias']:+.4f}  eff_res={metrics['eff_res_km']:.0f} km")

    df = pd.DataFrame(rows).sort_values(['var', 'depth_m'])
    df.to_csv(output_csv, index=False)
    print(f"\nwrote {output_csv}")
    print("\nskill-vs-depth summary (median nRMSE per variable family):")
    print(df.groupby('var')[['nrmse', 'eff_res_km']].median().to_markdown())
    return df


def _cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--truth-multidepth', required=True)
    p.add_argument('--truth-surface', required=True)
    p.add_argument('--time-start', default='2019-01-01')
    p.add_argument('--time-end', default='2019-12-31')
    p.add_argument('--dx-km', type=float, default=8.0, help='grid spacing in km (1/12 deg ~ 8 km at 40N)')
    p.add_argument('--manifest', default=None)
    p.add_argument('--output-csv', default='depth_metrics.csv')
    args = p.parse_args()
    main(args.test_dir, args.truth_multidepth, args.truth_surface,
         args.time_start, args.time_end, args.output_csv, manifest=args.manifest, dx_km=args.dx_km)


if __name__ == '__main__':
    _cli()
