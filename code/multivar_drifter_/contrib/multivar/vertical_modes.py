"""
Vertical-mode output head: instead of predicting each depth level as an
independent channel, the network predicts K coefficients per (variable
group, time step) which a FIXED linear projection (EOFs of the truth's
vertical profiles over the training period) maps back to the physical depth
levels.

Why (see the architecture discussion this implements): at ~21 depth levels
the truth profiles are massively redundant (a handful of EOFs captures most
of the variance of upper-ocean T/u/v profiles); predicting levels
independently wastes capacity relearning the same vertical structure 21
times and allows physically incoherent profiles. Predicting mode
coefficients divides the output dimension by ~2-4, guarantees smooth
plausible profiles, and encodes architecturally the fact that surface
observations constrain depth through coherent vertical structures.

Two pieces:
  - eofs_from_profiles / compute_vertical_eofs (numpy / CLI): EOF basis from
    the truth file, computed on PER-LEVEL STANDARDIZED training-period data
    so the projection lives in the same normalized space as the model's
    outputs (the datamodule normalizes each (var, depth) channel by its own
    train mean/std).
  - VerticalModesUNet (torch): drop-in replacement for the plain UNetModel
    solver (same forward contract: (B, C_in, H, W) -> (B, C_out, H, W)),
    selected via the `ablation=vertical_modes` config group.
"""
import argparse
import importlib

import numpy as np


# ----------------------------- EOF computation (numpy) -----------------------------

def eofs_from_profiles(profiles, n_modes):
    """
    profiles: (n_samples, n_levels) array of vertical profiles, finite rows only
        (standardize per level BEFORE calling if desired).
    Returns (components [n_modes, n_levels] - orthonormal rows, ordered by
    decreasing variance; explained_variance_ratio [n_modes]).
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    profiles = profiles[np.isfinite(profiles).all(axis=1)]
    if profiles.shape[0] < profiles.shape[1]:
        raise ValueError(f"need more samples ({profiles.shape[0]}) than levels ({profiles.shape[1]})")
    anom = profiles - profiles.mean(axis=0, keepdims=True)
    cov = anom.T @ anom / (anom.shape[0] - 1)
    eigval, eigvec = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    n_modes = min(n_modes, eigvec.shape[1])
    components = eigvec[:, :n_modes].T            # (n_modes, n_levels)
    evr = eigval[:n_modes] / eigval.sum()
    return components, evr


def compute_vertical_eofs(truth_path, truth_var, depth_indices, n_modes, output_npz,
                          time_slice=None, sample_stride=4, domain=None):
    """
    Build and save the standardized-EOF basis for one variable of the truth
    file, restricted to `depth_indices` (positions in the file's depth axis),
    optionally to a training `time_slice` ('2010-01-01', '2017-12-31') and a
    lat/lon `domain`. Saves: components, explained_variance_ratio, level_mean,
    level_std, depth_indices, depth_values.
    """
    import xarray as xr

    da = xr.open_dataset(truth_path)[truth_var]
    if 'latitude' in da.dims:
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = da.isel(depth=list(depth_indices))
    if time_slice is not None:
        da = da.sel(time=slice(*time_slice))
    if domain is not None:
        da = da.sel(lat=domain['lat'], lon=domain['lon'])
    da = da.isel(time=slice(None, None, sample_stride),
                 lat=slice(None, None, 2), lon=slice(None, None, 2))

    stacked = da.stack(sample=('time', 'lat', 'lon')).transpose('sample', 'depth').values
    stacked = stacked[np.isfinite(stacked).all(axis=1)]

    level_mean = stacked.mean(axis=0)
    level_std = stacked.std(axis=0)
    level_std = np.where(level_std > 0, level_std, 1.0)
    standardized = (stacked - level_mean) / level_std

    components, evr = eofs_from_profiles(standardized, n_modes)
    np.savez(output_npz, components=components, explained_variance_ratio=evr,
             level_mean=level_mean, level_std=level_std,
             depth_indices=np.asarray(depth_indices),
             depth_values=da['depth'].values)
    print(f"[eof] {truth_var}: {len(depth_indices)} levels -> {components.shape[0]} modes, "
          f"cumulative explained variance {evr.cumsum()[-1]:.4f} -> {output_npz}")
    return components, evr


# ----------------------------- torch head -----------------------------

def _lazy_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _Loaded:
    pass


def VerticalModesUNet(in_channels, multivar_dict, channels_per_dim, mode_specs,
                      **unet_kwargs):
    """
    Factory (Hydra _target_): shared UNetModel trunk predicting, per output
    group, either K x time mode-coefficient channels (groups listed in
    `mode_specs`) or the raw n_channels (pass-through groups, e.g. zos), then
    a fixed per-group projection back to the physical channel layout expected
    by the loss (multivar full_output order - contiguity enforced by
    get_multivar_head_groups).

    mode_specs: {group_name: {'npz': path, 'n_modes': K}}. The npz comes from
        compute_vertical_eofs and its number of levels must equal the group's
        channel count / channels_per_dim.
    """
    torch, nn = _lazy_torch()
    unet_kwargs.pop('out_channels', None)  # merged in by Hydra from the base solver config
    from contrib.multivar.multivar_utils import get_multivar_head_groups
    _unet_module = importlib.import_module('contrib.4dvarnet_latent.unet')
    UNetModel = _unet_module.UNetModel

    channel_groups = get_multivar_head_groups(multivar_dict, channels_per_dim)

    class _VerticalModesUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels_per_dim = channels_per_dim
            self.group_layout = []   # (name, trunk_channels, projection|None, n_levels)
            trunk_out = 0
            for name, n_channels in channel_groups.items():
                if name in mode_specs:
                    spec = mode_specs[name]
                    data = np.load(spec['npz'])
                    comp = data['components'][: int(spec['n_modes'])]     # (K, n_levels)
                    n_levels = n_channels // channels_per_dim
                    if comp.shape[1] != n_levels:
                        raise ValueError(
                            f"group '{name}': EOF basis has {comp.shape[1]} levels but the "
                            f"config declares {n_levels} depth channels - regenerate the npz "
                            f"with the same depth_indices as the fragments")
                    proj = torch.from_numpy(comp.T.astype(np.float32))    # (n_levels, K)
                    self.register_buffer(f'proj_{name}', proj)
                    k = comp.shape[0]
                    self.group_layout.append((name, k * channels_per_dim, f'proj_{name}', n_levels))
                    trunk_out += k * channels_per_dim
                else:
                    self.group_layout.append((name, n_channels, None, None))
                    trunk_out += n_channels
            self.trunk = UNetModel(in_channels=in_channels, out_channels=trunk_out, **unet_kwargs)

        def forward(self, x, timesteps=None, extra=None):
            h = self.trunk(x, timesteps=timesteps, extra=extra)
            outputs, start = [], 0
            for name, n_trunk, proj_name, n_levels in self.group_layout:
                block = h[:, start:start + n_trunk]
                start += n_trunk
                if proj_name is None:
                    outputs.append(block)
                else:
                    proj = getattr(self, proj_name)                      # (n_levels, K)
                    b, _, hh, ww = block.shape
                    coeffs = block.view(b, -1, self.channels_per_dim, hh, ww)   # (B, K, T, H, W)
                    levels = torch.einsum('lk,bkthw->blthw', proj, coeffs)
                    outputs.append(levels.reshape(b, n_levels * self.channels_per_dim, hh, ww))
            return torch.cat(outputs, dim=1)

    return _VerticalModesUNet()


def _cli():
    p = argparse.ArgumentParser(description="Compute the vertical EOF basis for the modes head")
    p.add_argument('--truth-path', required=True)
    p.add_argument('--truth-var', required=True)
    p.add_argument('--depth-indices', type=int, nargs='+', required=True)
    p.add_argument('--n-modes', type=int, default=8)
    p.add_argument('--train-start', default='2010-01-01')
    p.add_argument('--train-end', default='2017-12-31')
    p.add_argument('--sample-stride', type=int, default=4)
    p.add_argument('--output-npz', required=True)
    args = p.parse_args()
    compute_vertical_eofs(args.truth_path, args.truth_var, args.depth_indices, args.n_modes,
                          args.output_npz, time_slice=(args.train_start, args.train_end),
                          sample_stride=args.sample_stride)


if __name__ == '__main__':
    _cli()
