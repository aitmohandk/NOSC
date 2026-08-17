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
