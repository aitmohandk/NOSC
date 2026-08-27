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
