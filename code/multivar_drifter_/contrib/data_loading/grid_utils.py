"""Shared helpers for the common target grid used across the OSSE data pipeline.

The whole pipeline keeps a single source of truth for the working grid: the
prepared GLORYS file (produced by download_data_/prepare_glorys_osse.py, whose
--target-res sets the resolution). Every downstream producer - virtual ARGO,
observation masks, pseudo-obs - reads that grid via --grid-from / --truth-path,
so grid consistency is structural, not something each script re-derives.

`check_grid_resolution` is a light guard-rail against the one human error the
structure cannot prevent: pointing --grid-from at a GLORYS file that was NOT
regridded to the intended resolution (e.g. the native 1/12deg file). It only
runs when the caller passes an expected resolution; otherwise it does nothing.
"""
import numpy as np


def grid_step(coord_values):
    """Median absolute spacing of a 1-D coordinate array, in its own units."""
    v = np.asarray(coord_values, dtype=np.float64)
    if v.size < 2:
        return None
    return float(np.median(np.abs(np.diff(v))))


def check_grid_resolution(lat_grid, lon_grid, expected_res, tol=1e-3,
                          context="grid"):
    """Raise SystemExit if the (lat, lon) spacing differs from expected_res.

    No-op when expected_res is None. `tol` is the allowed absolute deviation in
    degrees (default 1e-3, i.e. one thousandth of a degree). Both axes are
    checked so a mismatch on either one is caught.
    """
    if expected_res is None:
        return
    dlat = grid_step(lat_grid)
    dlon = grid_step(lon_grid)
    problems = []
    if dlat is None or abs(dlat - expected_res) > tol:
        problems.append(f"lat spacing = {dlat}")
    if dlon is None or abs(dlon - expected_res) > tol:
        problems.append(f"lon spacing = {dlon}")
    if problems:
        raise SystemExit(
            f"[grid check] {context}: expected resolution {expected_res} deg but "
            f"found {', '.join(problems)} (tol {tol}). The reference grid does not "
            f"match --expect-res. Point --grid-from at the GLORYS file prepared "
            f"with the SAME --target-res, or drop --expect-res to skip this check."
        )
