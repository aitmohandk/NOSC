"""
Build the OSSE pseudo-observation file: truth (e.g. GLORYS zos) sampled along
the simulated satellite tracks, plus Gaussian instrument noise, written ONCE
to a NetCDF that the training config then treats as a plain input variable.

Why bake masking+noise into a file instead of using mask_path + on-the-fly
augmentation:
  - deterministic and identical across train/val/test (a real altimeter is
    noisy at test time too - evaluating on noise-free obs inflates skill);
  - the noise is in physical units, applied before normalization, once;
  - the training config stays a pure "input file -> target file" declaration
    with no hidden randomness.

Mask input formats:
  - .pickle : list of daily 2D arrays, INDEX-aligned with the truth's time
    axis (synthetic masks from build_masks with time_from=truth_path);
  - .nc     : a NetCDF with a `time` coordinate and a mask variable
    (auto-detected among: given mask_var, 'l3_mask', 'obs_mask', or the
    first data var), DATE-aligned via .sel - the right format for the REAL
    gridded L3 track masks (e.g. altimetry_traces/2010_2023/gridded/
    l3_mask.nc), which then give a historically faithful, non-stationary
    constellation instead of the synthetic stationary one.

The output file contains:
  - <var>_obs : truth + N(0, noise_std) where observed, NaN elsewhere
  - obs_mask  : 1.0 where observed, 0.0 elsewhere (explicit mask input channel)

Typical usage (Hydra entrypoint or CLI):
    _target_: contrib.synthetic_obs.make_pseudo_obs.make_pseudo_obs
    truth_path: <glorys.nc>
    truth_var: zos
    mask_path: <synthetic_masks.pickle>   # produced by build_masks with time_from=truth_path
    output_path: <pseudo_obs_ssh.nc>
    noise_std: 0.02                        # meters, typical nadir SLA noise ~2 cm
    seed: 1234
"""
import argparse
import pickle

import numpy as np
import xarray as xr


def make_pseudo_obs(truth_path, truth_var, mask_path, output_path, noise_std=0.0, seed=1234,
                    obs_var_name=None, lat_name='lat', lon_name='lon', skip_if_exists=True,
                    mask_var=None):
    """
    truth_path/truth_var: truth field (time, lat, lon), e.g. GLORYS 'zos'.
    mask_path: pickle list of daily masks (1.0/NaN), day 0 aligned with the
        truth file's first time step (generate with build_masks time_from=truth_path).
    noise_std: Gaussian instrument noise std in the truth variable's physical
        units (0 disables noise). Noise is drawn once, seeded, and saved -
        i.e. the same noisy obs are seen at every epoch and at test time,
        like real measurements.
    """
    from pathlib import Path as _P
    if skip_if_exists and _P(output_path).exists():
        print(f"[make_pseudo_obs] {output_path} exists - skipping (seeded/deterministic; "
              f"delete the file or pass skip_if_exists=False to force)")
        return output_path

    ds = xr.open_dataset(truth_path)
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': lat_name, 'longitude': lon_name})
    da = ds[truth_var]
    if 'depth' in da.dims:
        da = da.isel(depth=0)

    n_time = da.sizes['time']
    if str(mask_path).endswith(('.nc', '.nc4')):
        mds = xr.open_dataset(mask_path)
        if 'latitude' in mds.dims:
            mds = mds.rename({'latitude': lat_name, 'longitude': lon_name})
        cand = [v for v in ([mask_var] if mask_var else []) + ['l3_mask', 'obs_mask'] if v in mds]
        mvar = cand[0] if cand else list(mds.data_vars)[0]
        mda = mds[mvar].sel(time=da['time'], method='nearest', tolerance=np.timedelta64(1, 'D'))
        if (mda.sizes[lat_name], mda.sizes[lon_name]) != (da.sizes[lat_name], da.sizes[lon_name]):
            mda = mda.interp({lat_name: da[lat_name], lon_name: da[lon_name]}, method='nearest')
        raw = mda.values
        # accept either 1/NaN or 1/0 conventions
        observed = np.isfinite(raw) & (raw != 0)
    else:
        with open(mask_path, 'rb') as f:
            mask_list = pickle.load(f)
        masks = np.stack(mask_list, axis=0)
        if masks.shape[0] < n_time:
            raise ValueError(
                f"mask file covers {masks.shape[0]} days but truth has {n_time} time steps "
                f"(pickle masks are index-aligned: regenerate with build_masks time_from={truth_path}, "
                f"or use a date-aligned NetCDF mask)"
            )
        if masks.shape[1:] != (da.sizes[lat_name], da.sizes[lon_name]):
            raise ValueError(f"mask spatial shape {masks.shape[1:]} != truth {(da.sizes[lat_name], da.sizes[lon_name])}")
        observed = np.isfinite(masks[:n_time])
    values = da.values.astype(np.float32)

    if noise_std and noise_std > 0:
        rng = np.random.default_rng(seed)
        values = values + rng.normal(0.0, noise_std, size=values.shape).astype(np.float32)

    obs = np.where(observed, values, np.nan).astype(np.float32)
    obs_var_name = obs_var_name or f"{truth_var}_obs"

    out = xr.Dataset(
        {
            obs_var_name: (('time', lat_name, lon_name), obs),
            'obs_mask': (('time', lat_name, lon_name), observed.astype(np.float32)),
        },
        coords={c: da.coords[c] for c in ('time', lat_name, lon_name)},
        attrs=dict(
            source_truth=str(truth_path), source_var=str(truth_var),
            mask_file=str(mask_path), noise_std=float(noise_std or 0.0), noise_seed=int(seed),
        ),
    )
    out.to_netcdf(output_path)
    frac = float(observed.mean())
    print(f"[make_pseudo_obs] wrote {output_path}: {obs_var_name} + obs_mask, "
          f"{n_time} days, mean daily coverage {100 * frac:.2f}%")
    return output_path


def _cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--truth-path', required=True)
    p.add_argument('--truth-var', required=True)
    p.add_argument('--mask-path', required=True)
    p.add_argument('--output-path', required=True)
    p.add_argument('--noise-std', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    make_pseudo_obs(args.truth_path, args.truth_var, args.mask_path, args.output_path,
                    noise_std=args.noise_std, seed=args.seed)


if __name__ == '__main__':
    _cli()
