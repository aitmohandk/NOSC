#!/usr/bin/env python
"""Prepare the consolidated GLORYS files the OSSE configs expect.

The multivar OSSE configs (config/xp/osse3d_gs*.yaml) read GLORYS truth from two
*single* NetCDF files via ${paths.data_root}:

    glorys_gs_surface_<period>.nc      # surface fields (zos + level-0 thetao/uo/vo)
    glorys_gs_multidepth_<period>.nc   # full depth truth (thetao/uo/vo/zos, 5 levels)

The runtime data loader (contrib/data_loading/data.py :: open_var_dataset) opens
ONE file per var_path with xr.open_dataset (no glob), renames the CF coordinates
latitude/longitude itself, and subsets the spatial domain and depth level lazily.
So the prepared files only need to be a single consolidated NetCDF each, with the
native GLORYS variable names (thetao, uo, vo, zos) and CF coordinates
(latitude, longitude, depth, time). This script builds exactly those two files.

Crucially, the *source* of the raw GLORYS is decoupled from where the prepared
files are written, so the SAME command works whether the raw data was:

  * downloaded with copernicusmarine (import_data_glorys_multidepth.py), landing
    in $NOSC_DATA_ROOT/glorys_raw, or
  * already mirrored on Datarmor under the reference-data space, e.g.
    /home/ref-cmems/... or /home/ref-ocean-reanalysis/... (read-only) - in which
    case you download nothing and just point --src at that directory.

Only the value of --src (or $NOSC_GLORYS_SRC) changes between the two cases.

Examples
--------
# From a copernicusmarine download (default source = $NOSC_DATA_ROOT/glorys_raw):
export NOSC_DATA_ROOT=$DATAWORK/nosc/data
python prepare_glorys_osse.py

# Straight from Datarmor reference data, no download:
export NOSC_DATA_ROOT=$DATAWORK/nosc/data
python prepare_glorys_osse.py --src /home/ref-ocean-reanalysis/<...>/glorys_native
"""
import argparse
import glob as _glob
import os
import re
import sys

import xarray as xr


# Gulf Stream OSSE defaults (match config/xp/osse3d_gs*.yaml).
DEF_LAT = (32.0, 44.0)
DEF_LON = (-66.0, -54.0)
DEF_START = "2010-01-01"
DEF_END = "2020-01-01"
GLORYS_VARS = ["thetao", "uo", "vo", "zos"]
# The 5 OSSE depth levels (config/vars/*_depths.yaml). GLORYS ships 50 native
# levels; keeping only these 5 shrinks the multidepth file ~10x. The runtime
# loader selects by nearest depth anyway, so keeping the native values here is
# exact.
DEF_DEPTHS = [0.494, 15.0, 50.0, 100.0, 200.0]
_YEAR_RE = re.compile(r"^\d{4}$")


def _env_default(name, fallback=None):
    val = os.environ.get(name)
    return val if val else fallback


def parse_args(argv=None):
    data_root = _env_default("NOSC_DATA_ROOT")
    src_default = _env_default("NOSC_GLORYS_SRC")
    if src_default is None and data_root is not None:
        src_default = os.path.join(data_root, "glorys_raw")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=src_default,
                   help="Raw GLORYS source: a directory (searched recursively, and "
                        "restricted to the requested years when it is laid out as "
                        "YYYY/MM/*.nc like the Datarmor reference mirror) or a single "
                        ".nc file. Defaults to $NOSC_GLORYS_SRC, else "
                        "$NOSC_DATA_ROOT/glorys_raw.")
    p.add_argument("--out", default=data_root,
                   help="Output directory for the consolidated files. Defaults to "
                        "$NOSC_DATA_ROOT (a durable, non-purged space).")
    p.add_argument("--lat-min", type=float, default=DEF_LAT[0])
    p.add_argument("--lat-max", type=float, default=DEF_LAT[1])
    p.add_argument("--lon-min", type=float, default=DEF_LON[0])
    p.add_argument("--lon-max", type=float, default=DEF_LON[1])
    p.add_argument("--start", default=DEF_START, help="Start date (inclusive), YYYY-MM-DD.")
    p.add_argument("--end", default=DEF_END, help="End date (exclusive-ish), YYYY-MM-DD.")
    p.add_argument("--period-label", default=None,
                   help="Filename suffix; default derived from --start/--end years "
                        "(e.g. 2010-2020) to match the config's expected names.")
    p.add_argument("--depths", type=float, nargs="+", default=DEF_DEPTHS,
                   help="Depth levels to keep (nearest match), default the 5 OSSE "
                        "levels. GLORYS ships 50 native levels; keeping 5 shrinks the "
                        "multidepth file ~10x.")
    p.add_argument("--keep-all-depths", action="store_true",
                   help="Keep all native depth levels instead of --depths.")
    p.add_argument("--no-float32", action="store_true",
                   help="Keep native float64 instead of downcasting to float32 "
                        "(float32 roughly halves the output size).")
    p.add_argument("--glob", default="*.nc",
                   help="Filename pattern matched at any depth under --src.")
    return p.parse_args(argv)


def _coord_names(ds):
    lat = "latitude" if ("latitude" in ds.coords or "latitude" in ds.dims) else "lat"
    lon = "longitude" if ("longitude" in ds.coords or "longitude" in ds.dims) else "lon"
    return lat, lon


def _spatial_subset(ds, args):
    """Subset one dataset to the lon/lat box, robust to descending latitude and
    to the latitude/longitude vs lat/lon naming. Used as open_mfdataset's
    per-file preprocess so each global file is cut to the box before
    concatenation (keeps memory low)."""
    lat, lon = _coord_names(ds)
    lat_lo, lat_hi = args.lat_min, args.lat_max
    if float(ds[lat][0]) > float(ds[lat][-1]):
        lat_lo, lat_hi = lat_hi, lat_lo
    lon_lo, lon_hi = args.lon_min, args.lon_max
    if float(ds[lon][0]) > float(ds[lon][-1]):
        lon_lo, lon_hi = lon_hi, lon_lo
    return ds.sel({lat: slice(lat_lo, lat_hi), lon: slice(lon_lo, lon_hi)})


def list_source_files(src, pattern, start, end):
    """Return the list of .nc files under a directory source.

    If src is laid out as YYYY/ subdirectories (the Datarmor reference mirror:
    global-reanalysis-.../<year>/<month>/*.nc), only the years overlapping
    [start, end] are descended into - opening the full 30+ year global archive
    just to subset would be pointlessly slow. Otherwise the whole tree is
    searched recursively (covers the flat glorys_raw/ download layout too).
    """
    y0, y1 = int(start[:4]), int(end[:4])
    year_dirs = sorted(d for d in os.listdir(src)
                       if _YEAR_RE.match(d) and os.path.isdir(os.path.join(src, d)))
    files = []
    if year_dirs:
        for d in year_dirs:
            if y0 <= int(d) <= y1:
                files += _glob.glob(os.path.join(src, d, "**", pattern), recursive=True)
    else:
        files = _glob.glob(os.path.join(src, "**", pattern), recursive=True)
    return sorted(files)


def open_source(src, args):
    """Open the raw GLORYS source (a dir tree or a single file), spatially
    subset to the box (per-file for directories), then subset time."""
    if src is None:
        sys.exit("error: no source given. Set --src or $NOSC_GLORYS_SRC, or "
                 "$NOSC_DATA_ROOT (defaulting --src to $NOSC_DATA_ROOT/glorys_raw).")

    if os.path.isdir(src):
        files = list_source_files(src, args.glob, args.start, args.end)
        if not files:
            sys.exit(f"error: no files matching '{args.glob}' under {src} "
                     f"for years {args.start[:4]}-{args.end[:4]}.")
        print(f"[prepare_glorys_osse] {len(files)} fichier(s) source à ouvrir")
        ds = xr.open_mfdataset(
            files, combine="by_coords", chunks={"time": 30},
            preprocess=lambda d: _spatial_subset(d, args),
        )
    elif os.path.isfile(src):
        ds = _spatial_subset(xr.open_dataset(src, chunks={"time": 30}), args)
    else:
        sys.exit(f"error: source not found (neither dir nor file): {src}")

    if "time" in ds.dims or "time" in ds.coords:
        ds = ds.sel(time=slice(args.start, args.end))
    return ds


def main(argv=None):
    args = parse_args(argv)
    if not args.out:
        sys.exit("error: no output dir. Set $NOSC_DATA_ROOT or pass --out.")
    os.makedirs(args.out, exist_ok=True)

    label = args.period_label or f"{args.start[:4]}-{args.end[:4]}"

    print(f"[prepare_glorys_osse] source : {args.src}")
    print(f"[prepare_glorys_osse] output : {args.out}")
    print(f"[prepare_glorys_osse] domain : lat[{args.lat_min},{args.lat_max}] "
          f"lon[{args.lon_min},{args.lon_max}] time[{args.start},{args.end}]")

    ds = open_source(args.src, args)
    present = [v for v in GLORYS_VARS if v in ds.variables]
    missing = [v for v in GLORYS_VARS if v not in ds.variables]
    if missing:
        print(f"[prepare_glorys_osse] WARNING: variables absentes de la source, "
              f"ignorées : {missing}")
    ds = ds[present]

    # Keep only the OSSE depth levels (nearest match) unless asked otherwise:
    # GLORYS has 50 native levels, the OSSE config uses 5 -> ~10x smaller.
    if "depth" in ds.dims and not args.keep_all_depths:
        ds = ds.sel(depth=args.depths, method="nearest")

    # Downcast to float32 (GLORYS ships float64; halves the size, precision is
    # ample for this use).
    if not args.no_float32:
        ds = ds.astype("float32")

    surface_path = os.path.join(args.out, f"glorys_gs_surface_{label}.nc")
    multidepth_path = os.path.join(args.out, f"glorys_gs_multidepth_{label}.nc")

    # zlib compression on every data variable.
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}

    # multidepth truth: the selected depth levels.
    print(f"[prepare_glorys_osse] écriture {multidepth_path} ...")
    ds.to_netcdf(multidepth_path, encoding=enc)

    # surface file: level-0 slice (zos has no depth dim and is preserved).
    surface = ds.isel(depth=0) if "depth" in ds.dims else ds
    enc_s = {v: {"zlib": True, "complevel": 4} for v in surface.data_vars}
    print(f"[prepare_glorys_osse] écriture {surface_path} ...")
    surface.to_netcdf(surface_path, encoding=enc_s)

    print("[prepare_glorys_osse] terminé.")


if __name__ == "__main__":
    main()
