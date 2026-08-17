"""
Config-authoring helper for the "variable x depth" extension of the multivar
mechanism (Phase 2 of the NOSC architecture redesign, see
staged-meandering-ocean.md). Turns one variable template + a list of depth
levels into one multivar-dict entry per (variable, depth) pair - each entry
becomes its own stacked channel group at runtime via the existing, unmodified
open_multivar_datasets / MultivarBatchSelector mechanism.

This is a codegen convenience only: it does not run inside the Hydra compose
step. Use it once to write a config/vars/*.yaml fragment, then reference that
fragment from an xp config's `defaults:` list.
"""
from pathlib import Path

import yaml


def _depth_str(depth):
    if isinstance(depth, float) and depth.is_integer():
        depth = int(depth)
    return str(depth)


def expand_depth_levels(name, template, depth_levels, unit='m'):
    """
    name: base variable name, e.g. "thetao".
    template: shared var_info fields (var_path, var_name, input_arch,
        output_arch, broadcast_time, ...); must NOT already set depth_level.
    depth_levels: depth values in the source dataset's `depth` coordinate
        units (typically meters); nearest available level is selected at
        load time by open_var_dataset.

    Returns {f"{name}_{depth}{unit}": {**template, "depth_level": depth}, ...}
    """
    if 'depth_level' in template:
        raise ValueError("template must not set depth_level - it is added per depth entry")

    return {
        f"{name}_{_depth_str(depth)}{unit}": {**template, "depth_level": depth}
        for depth in depth_levels
    }


def write_depth_levels_yaml(output_path, name, template, depth_levels, package="multivar", unit='m'):
    """
    Write a Hydra config-group fragment at `output_path` under `@package
    {package}` (default "multivar"), so its entries merge straight into an xp
    config's `multivar:` dict via a `defaults:` entry, e.g. with this file
    saved at config/vars/thetao_depths.yaml:

        defaults:
          - vars/thetao_depths
          - _self_
    """
    entries = expand_depth_levels(name, template, depth_levels, unit=unit)
    header = f"# @package {package}\n"
    body = yaml.safe_dump(entries, sort_keys=False)
    Path(output_path).write_text(header + body)
    return entries
