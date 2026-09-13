"""
Argo float profile download, via argopy (https://argopy.readthedocs.io).

argopy is imported lazily inside the functions below (not at module import
time), following this repo's existing convention for optional heavy
dependencies (see e.g. cosanneal_lr_lion's `import lion_pytorch` in
src/utils.py) - argopy is not part of env/4dvarnet-daniel.yaml today.
"""


def fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                         min_depth=0, max_depth=2000, mode='standard'):
    """
    Fetch raw Argo profiles (core variables: PRES, TEMP, PSAL + QC flags) for
    a region/period as an xarray.Dataset in argopy's point-cloud
    (N_POINTS) layout.

    mode: 'standard' (delayed-mode QC'd where available) or 'expert' (all
        profiles, unfiltered by argopy's own qc pre-selection - use qc.py to
        filter explicitly instead).
    """
    import argopy

    argopy.set_options(mode=mode)
    fetcher = argopy.DataFetcher(mode=mode).region(
        [lon_min, lon_max, lat_min, lat_max, min_depth, max_depth, start_date, end_date]
    )
    return fetcher.to_xarray()


def fetch_argo_profiles_for_domain(domain, start_date, end_date, **kwargs):
    """domain: dict with lat/lon slice() values, e.g. the `domain` config group
    used throughout this repo's xp configs ({'lat': slice(-70, 70), 'lon': slice(-180, 180)})."""
    lat_slice, lon_slice = domain['lat'], domain['lon']
    return fetch_argo_profiles(
        lon_slice.start, lon_slice.stop, lat_slice.start, lat_slice.stop,
        start_date, end_date, **kwargs,
    )


def fetch_argo_profiles_chunked(lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                                 freq='MS', min_depth=0, max_depth=2000, mode='standard',
                                 on_chunk_error='warn'):
    """
    Same as fetch_argo_profiles, but fetched in consecutive time chunks
    (default: month starts, 'MS') and concatenated. A single region request
    for a decade of global profiles (millions of points) times out or
    exhausts memory on the erddap backend - decade-scale pulls must be
    chunked.

    on_chunk_error: 'raise' or 'warn' (skip the failing chunk with a warning;
        useful for long unattended pulls where one transient backend error
        should not lose hours of progress).
    """
    import pandas as pd
    import xarray as xr

    edges = pd.date_range(start_date, end_date, freq=freq)
    if len(edges) == 0 or edges[0] > pd.Timestamp(start_date):
        edges = edges.insert(0, pd.Timestamp(start_date))
    if edges[-1] < pd.Timestamp(end_date):
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    chunks = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        try:
            chunk = fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max,
                                        str(t0.date()), str(t1.date()),
                                        min_depth=min_depth, max_depth=max_depth, mode=mode)
            chunks.append(chunk)
            print(f"[argo] fetched {t0.date()}..{t1.date()}: "
                  f"{chunk.sizes.get('N_POINTS', chunk.sizes.get('N_PROF', 0))} points")
        except Exception as exc:
            if on_chunk_error == 'raise':
                raise
            print(f"[argo] WARNING: chunk {t0.date()}..{t1.date()} failed ({exc}); skipping")

    if not chunks:
        raise RuntimeError("no ARGO chunk could be fetched")
    dim = 'N_POINTS' if 'N_POINTS' in chunks[0].dims else 'N_PROF'
    return xr.concat(chunks, dim=dim)


# ---------------------------------------------------------------------------
# Local GDAC source (alternative to argopy / no Internet).
#
# On Datarmor the native Argo GDAC is mirrored read-only (e.g. /home/ref-argo/
# gdac). Reading it directly avoids the argopy/erddapy dependency and the ftp
# queue entirely. These functions return the SAME point-cloud (N_POINTS) layout
# as fetch_argo_profiles above - PRES/TEMP/PSAL + *_QC + LATITUDE/LONGITUDE/JULD
# + PLATFORM_NUMBER - so the downstream pipeline (qc.apply_standard_qc,
# interp_argo_profiles, virtualize_profiles) is unchanged.
# ---------------------------------------------------------------------------

# GDAC per-profile files store one or more profiles with dims (N_PROF, N_LEVELS).
# We flatten to the N_POINTS point cloud argopy produces. QC flag chars ('1'..)
# are decoded to ints; argopy exposes them as small ints, so we match that.
_GDAC_VALUE_VARS = ("TEMP", "PSAL")


def _decode_qc(arr):
    """GDAC QC flags are single bytes/chars ('0'..'9', b' '); argopy yields ints.
    Convert to int, mapping blanks/fill to 9 (bad/missing)."""
    import numpy as np
    out = np.full(arr.shape, 9, dtype="int8")
    flat = np.asarray(arr).ravel()
    res = out.ravel()
    for i, v in enumerate(flat):
        if isinstance(v, bytes):
            v = v.decode("ascii", "ignore")
        v = str(v).strip()
        if v.isdigit():
            res[i] = int(v)
    return out


def _profile_file_to_pointcloud(ds, value_vars=_GDAC_VALUE_VARS):
    """Flatten one GDAC profile dataset (N_PROF, N_LEVELS) to an N_POINTS cloud
    with the variables the QC/interp pipeline expects."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    n_prof = ds.sizes.get("N_PROF", 1)
    n_lev = ds.sizes.get("N_LEVELS", ds.sizes.get("N_LEVEL", 1))

    def col(name, per_level):
        if name not in ds.variables:
            return None
        a = np.asarray(ds[name].values)
        if per_level:
            a = np.broadcast_to(a.reshape(n_prof, -1)[:, :n_lev], (n_prof, n_lev))
            return a.reshape(-1)
        return np.repeat(a.reshape(n_prof)[:n_prof], n_lev)

    juld = ds["JULD"].values  # datetime64 in argo files (reference 1950 handled by xarray)
    data = {
        "PRES": ("N_POINTS", col("PRES", True)),
        "LATITUDE": ("N_POINTS", col("LATITUDE", False)),
        "LONGITUDE": ("N_POINTS", col("LONGITUDE", False)),
        "JULD": ("N_POINTS", np.repeat(np.asarray(juld).reshape(n_prof)[:n_prof], n_lev)),
        "PRES_QC": ("N_POINTS", _decode_qc(col("PRES_QC", True))),
        "POSITION_QC": ("N_POINTS", _decode_qc(col("POSITION_QC", False))),
        "JULD_QC": ("N_POINTS", _decode_qc(col("JULD_QC", False))),
    }
    if "PLATFORM_NUMBER" in ds.variables:
        plat = np.asarray(ds["PLATFORM_NUMBER"].values).reshape(n_prof)
        data["PLATFORM_NUMBER"] = ("N_POINTS", np.repeat(plat[:n_prof], n_lev))
    # CYCLE_NUMBER is used by qc.sort_pointcloud to order points within a
    # profile; argopy provides it. Fall back to a per-profile index if absent.
    if "CYCLE_NUMBER" in ds.variables:
        cyc = np.asarray(ds["CYCLE_NUMBER"].values).reshape(n_prof)
    else:
        cyc = np.arange(n_prof)
    data["CYCLE_NUMBER"] = ("N_POINTS", np.repeat(cyc[:n_prof], n_lev))
    for vv in value_vars:
        c = col(vv, True)
        if c is not None:
            data[vv] = ("N_POINTS", c)
            data[f"{vv}_QC"] = ("N_POINTS", _decode_qc(col(f"{vv}_QC", True)))
    return xr.Dataset(data)


def fetch_argo_profiles_local(gdac_dir, lon_min, lon_max, lat_min, lat_max,
                              start_date, end_date, min_depth=0, max_depth=2000,
                              value_vars=_GDAC_VALUE_VARS, file_glob="**/*.nc",
                              on_file_error="warn", **_ignored):
    """Build the argopy-style N_POINTS point cloud from a local GDAC mirror.

    gdac_dir: root of the GDAC tree (e.g. /home/ref-argo/gdac). All NetCDF
        profile files matching file_glob are scanned; profiles are kept if
        their position falls in the box and their time in [start, end].
    Returns an xarray.Dataset with the same layout as fetch_argo_profiles, so
    it is a drop-in replacement feeding qc.apply_standard_qc.

    Extra keyword args are ignored, so this can be called with the same
    signature as the argopy fetchers (mode=, freq=, ...).
    """
    import glob as _glob
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr

    t0 = pd.Timestamp(start_date)
    t1 = pd.Timestamp(end_date)
    files = sorted(_glob.glob(os.path.join(gdac_dir, file_glob), recursive=True))
    if not files:
        raise RuntimeError(f"no GDAC profile files under {gdac_dir} (glob {file_glob})")

    clouds = []
    for fp in files:
        try:
            with xr.open_dataset(fp, decode_times=True) as ds:
                pc = _profile_file_to_pointcloud(ds, value_vars=value_vars)
        except Exception as exc:
            if on_file_error == "raise":
                raise
            print(f"[argo-local] WARNING: {fp} failed ({exc}); skipping")
            continue
        lat = pc["LATITUDE"].values
        lon = pc["LONGITUDE"].values
        pres = pc["PRES"].values
        t = pd.to_datetime(pc["JULD"].values)
        keep = (
            (lat >= lat_min) & (lat <= lat_max)
            & (lon >= lon_min) & (lon <= lon_max)
            & (pres >= min_depth) & (pres <= max_depth)
            & (t >= t0) & (t <= t1)
        )
        if keep.any():
            clouds.append(pc.isel(N_POINTS=np.where(keep)[0]))

    if not clouds:
        raise RuntimeError(
            f"no ARGO profile in box/period from {gdac_dir} "
            f"(lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}] "
            f"time[{start_date},{end_date}])")
    out = xr.concat(clouds, dim="N_POINTS")
    print(f"[argo-local] {len(clouds)} file(s), {out.sizes['N_POINTS']} points in box/period")
    return out
