"""
Shared trunk + per-variable-group heads, as an alternative solver to plain
MultivarUNet_mae's single shared output convolution
(contrib.4dvarnet_latent.unet.UNetModel.out: one joint Conv2d producing all
output channels at once, see the conversation this module was written for).

Rationale (multi-task learning literature: Caruana 1997, Ruder 2017 survey,
Kendall et al. 2018): output channels here are physically heterogeneous
(SSH in meters, temperature in degC, currents in m/s, at multiple depths and
observation densities), which is exactly the setting where a single shared
final projection risks gradient conflict between tasks. This module does not
modify contrib/4dvarnet_latent/unet.py (a vendored generic architecture) -
it reuses UNetModel unchanged as a shared trunk down to a modest shared
"neck" width, then attaches small independent conv heads per group.
"""
import importlib

import torch
import torch.nn as nn

from contrib.multivar.multivar_utils import get_multivar_head_groups

# contrib.4dvarnet_latent is not a valid Python dotted-import path (starts
# with a digit) - the rest of this codebase only ever reaches it through
# Hydra's string-based _target_ instantiation; do the same here.
_unet_module = importlib.import_module('contrib.4dvarnet_latent.unet')
UNetModel = _unet_module.UNetModel
conv_nd = _unet_module.conv_nd
zero_module = _unet_module.zero_module


class GroupedHeads(nn.Module):
    """One independent small conv stack per group, applied to the same shared
    feature map, concatenated back in group order (== multivar full_output order,
    see get_multivar_head_groups's contiguity requirement)."""

    def __init__(self, in_channels, channel_groups, hidden_channels=32, kernel_size=3, dims=2, n_layers=2):
        super().__init__()
        self.channel_groups = channel_groups
        self.heads = nn.ModuleDict()
        for name, n_out in channel_groups.items():
            layers = []
            c_in = in_channels
            for _ in range(max(n_layers - 1, 0)):
                layers += [conv_nd(dims, c_in, hidden_channels, kernel_size, padding=kernel_size // 2), nn.SiLU()]
                c_in = hidden_channels
            layers.append(zero_module(conv_nd(dims, c_in, n_out, kernel_size, padding=kernel_size // 2)))
            self.heads[name] = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat([self.heads[name](x) for name in self.channel_groups], dim=1)


class HeadedMultivarUNet(nn.Module):
    """
    trunk: UNetModel(..., out_channels=neck_channels) - unmodified, acting as
        a shared feature extractor down to a shared bottleneck width.
    heads: GroupedHeads(neck_channels -> per-group output channels).
    """

    def __init__(
        self, in_channels, multivar_dict, channels_per_dim,
        neck_channels=32, head_hidden_channels=32, head_layers=2, default_head_group=None,
        **unet_kwargs,
    ):
        super().__init__()
        # Hydra config-group overrides MERGE with the base solver config: an
        # inherited out_channels key would collide with our own neck_channels
        # wiring - discard it explicitly.
        unet_kwargs.pop('out_channels', None)
        self.channel_groups = get_multivar_head_groups(multivar_dict, channels_per_dim, default_group=default_head_group)
        self.trunk = UNetModel(in_channels=in_channels, out_channels=neck_channels, **unet_kwargs)
        self.heads = GroupedHeads(
            neck_channels, self.channel_groups, hidden_channels=head_hidden_channels,
            n_layers=head_layers, dims=unet_kwargs.get('dims', 2),
        )

    def forward(self, x, timesteps=None, extra=None):
        shared = self.trunk(x, timesteps=timesteps, extra=extra)
        return self.heads(shared)
