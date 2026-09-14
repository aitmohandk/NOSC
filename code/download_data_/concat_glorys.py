#!/usr/bin/env python3
"""Merge the per-year GLORYS files produced by prepare_glorys.pbs into the two
consolidated files the OSSE configs expect.

Kept as a real .py file (not an inline `python -c "..."`) on purpose: the PBS
jobs run under csh, and csh cannot carry a multi-line double-quoted argument,
so an inline program breaks with `Unmatched "`. A file sidesteps all quoting.

Usage:
    python -u concat_glorys.py <by_year_dir> <out_dir> [label]

<by_year_dir>  directory holding glorys_gs_{surface,multidepth}_<year>.nc
<out_dir>      where the merged files are written (created if missing)
label          suffix for the merged files (default: 2010-2020)
"""
import glob
import os
import sys

# Line-buffer stdout so progress lands in the PBS .o file as it happens, even
# if the job is killed on walltime (same reason as in prepare_glorys_osse.py).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import xarray as xr


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        sys.exit("usage: concat_glorys.py <by_year_dir> <out_dir> [label]")
    by_year, out = argv[0], argv[1]
    label = argv[2] if len(argv) > 2 else "2010-2020"

    os.makedirs(out, exist_ok=True)

    for kind in ("surface", "multidepth"):
        pattern = os.path.join(by_year, f"glorys_gs_{kind}_[0-9][0-9][0-9][0-9].nc")
        files = sorted(glob.glob(pattern))
        if not files:
            sys.exit(f"error: no per-year {kind} files matching {pattern}")
        print(f"[concat] {kind}: {len(files)} fichier(s) -> fusion", flush=True)
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 30})
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        path = os.path.join(out, f"glorys_gs_{kind}_{label}.nc")
        ds.to_netcdf(path, encoding=enc)
        print(f"[concat] écrit {path}", flush=True)

    print("[concat] terminé.", flush=True)


if __name__ == "__main__":
    main()
