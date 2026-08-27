"""
Framework-agnostic combination of per-output-variable losses into a scalar,
with optional grouping (pure Python on any objects supporting + and *, so it
works on torch scalars in training and on floats in unit tests).

Modes:
  flat_sum   : sum of all per-variable losses (historical behaviour). With
               many depth channels this over-weights 3D variables: at 21
               levels x {thetao, uo, vo} + zos, SSH carries 1/64 of the
               gradient and the current channels 42/64.
  group_mean : mean within each group, then weighted sum across groups
               (default weight 1 per group) - each physical quantity
               contributes equally regardless of how many depth channels
               represent it. Groups come from the multivar entries'
               `head_group` tags (get_multivar_loss_groups); an untagged
               variable is its own group.
"""


def get_multivar_loss_groups(multivar_dict, default_group=None):
    """{output var name -> group name} for full_output entries, from head_group
    tags (same tags the heads architecture uses; contiguity NOT required here)."""
    groups = {}
    for var, var_info in multivar_dict.items():
        if var_info.output_arch != 'full_output':
            continue
        groups[var] = var_info.get('head_group', default_group) or var
    return groups


def combine_grouped_losses(per_var_losses, output_var_names, loss_groups=None,
                            mode='flat_sum', group_weights=None):
    """
    per_var_losses: list of loss scalars, aligned with output_var_names.
    loss_groups: {var_name: group}; required for mode='group_mean'.
    group_weights: optional {group: weight} (default 1.0 per group).
    Returns (total_loss, {group: group_loss_after_weighting}).
    """
    if mode not in ('flat_sum', 'group_mean'):
        raise ValueError(f"unknown loss combination mode '{mode}'")

    if mode == 'flat_sum':
        total = None
        for loss in per_var_losses:
            total = loss if total is None else total + loss
        return total, {'all': total}

    if loss_groups is None:
        raise ValueError("mode='group_mean' requires loss_groups")
    group_weights = group_weights or {}

    sums, counts = {}, {}
    for name, loss in zip(output_var_names, per_var_losses):
        group = loss_groups.get(name, name)
        sums[group] = loss if group not in sums else sums[group] + loss
        counts[group] = counts.get(group, 0) + 1

    total = None
    weighted = {}
    for group, s in sums.items():
        g = float(group_weights.get(group, 1.0)) * (1.0 / counts[group]) * s
        weighted[group] = g
        total = g if total is None else total + g
    return total, weighted
