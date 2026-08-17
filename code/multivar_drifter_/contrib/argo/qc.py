"""
Argo quality-control filtering, following the standard Argo QC flag
convention (1=good, 2=probably good, 3=probably bad, 4=bad, 5=changed,
8=estimated, 9=missing - see the Argo User's Manual). Operates on an
xarray.Dataset in argopy's point-cloud (N_POINTS) layout, i.e. one row per
(profile, level).
"""
import numpy as np
import xarray as xr

GOOD_FLAGS = (1, 2)


def _qc_as_int(qc_da):
    """Argo QC flags are stored as single-character bytes/strings ('1', b'1', ...)."""
    values = qc_da.values
    if values.dtype.kind in ('S', 'U', 'O'):
        values = np.array([int(v) if str(v).strip() not in ('', 'nan') else 9 for v in values.astype(str)])
    return values.astype(int)


def _profile_key(ds, profile_id_var, cycle_var):
    """Per-point profile identifier, as an object array (numpy's fixed-width
    unicode dtype doesn't support elementwise `+` across differing widths)."""
    platform = ds[profile_id_var].astype(str).values.astype(object)
    cycle = ds[cycle_var].astype(str).values.astype(object)
    return platform + '_' + cycle


def filter_by_qc(ds: xr.Dataset, qc_vars, good_flags=GOOD_FLAGS):
    """Keep only points where every QC variable in `qc_vars` is in `good_flags`."""
    keep = np.ones(ds.sizes.get('N_POINTS', ds.sizes.get('N_PROF')), dtype=bool)
    for qc_var in qc_vars:
        if qc_var in ds:
            keep &= np.isin(_qc_as_int(ds[qc_var]), good_flags)
    dim = 'N_POINTS' if 'N_POINTS' in ds.dims else 'N_PROF'
    return ds.isel({dim: keep})


def reject_pressure_inversions(ds: xr.Dataset, pres_var='PRES', profile_id_var='PLATFORM_NUMBER',
                                cycle_var='CYCLE_NUMBER'):
    """
    Drop points where PRES is not (weakly) monotonically increasing within a
    profile - a standard Argo QC test for pressure inversions/duplicates.
    Assumes ds is already sorted by (profile_id, cycle, pres); if not, the
    caller should sort first.
    """
    dim = 'N_POINTS' if 'N_POINTS' in ds.dims else 'N_PROF'
    profile_key = _profile_key(ds, profile_id_var, cycle_var)
    pres = ds[pres_var].values

    keep = np.ones(ds.sizes[dim], dtype=bool)
    same_profile = profile_key[1:] == profile_key[:-1]
    non_increasing = pres[1:] <= pres[:-1]
    keep[1:] &= ~(same_profile & non_increasing)

    return ds.isel({dim: keep})


def reject_spikes(ds: xr.Dataset, value_var, threshold, pres_var='PRES', profile_id_var='PLATFORM_NUMBER',
                   cycle_var='CYCLE_NUMBER'):
    """
    Simple vertical-spike test: flag a point as a spike if it deviates from
    the average of its two vertical neighbors (within the same profile) by
    more than `threshold`, matching the spirit of the Argo QC spike test
    (a lightweight version - not the full Argo delayed-mode algorithm).
    """
    dim = 'N_POINTS' if 'N_POINTS' in ds.dims else 'N_PROF'
    profile_key = _profile_key(ds, profile_id_var, cycle_var)
    values = ds[value_var].values

    is_spike = np.zeros(ds.sizes[dim], dtype=bool)
    interior = np.arange(1, len(values) - 1)
    same_profile = (profile_key[interior - 1] == profile_key[interior]) & (profile_key[interior + 1] == profile_key[interior])
    neighbor_avg = (values[interior - 1] + values[interior + 1]) / 2.0
    deviation = np.abs(values[interior] - neighbor_avg)
    is_spike[interior] = same_profile & (deviation > threshold)

    return ds.isel({dim: ~is_spike})


def apply_standard_qc(ds: xr.Dataset, value_vars=('TEMP', 'PSAL'), spike_thresholds=None):
    """
    Convenience pipeline: position/time QC -> per-variable QC -> pressure
    inversions -> (optional) spike rejection.

    spike_thresholds: optional dict {value_var: threshold} to also run
        reject_spikes for that variable.
    """
    qc_vars = ['POSITION_QC', 'JULD_QC', 'PRES_QC'] + [f'{v}_QC' for v in value_vars]
    ds = filter_by_qc(ds, qc_vars)
    ds = reject_pressure_inversions(ds)

    if spike_thresholds:
        for value_var, threshold in spike_thresholds.items():
            ds = reject_spikes(ds, value_var, threshold)

    return ds
