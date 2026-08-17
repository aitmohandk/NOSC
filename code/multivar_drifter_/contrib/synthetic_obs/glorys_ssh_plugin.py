"""
swot_simulator (CNES/JPL, https://github.com/CNES/swot_simulator) ssh_plugin
backed by NOSC's Glorys SLA/SSH source file, following the same
"single multi-year file, lon/lat/time dims" convention already used by
open_var_dataset (contrib/data_loading/data.py) - so the same var_path
already declared in a multivar: entry can be reused here directly.

Mirrors swot_simulator.plugins.ssh.aviso.AVISO (the bundled CMEMS L4 plugin),
swapped to NOSC's Glorys file layout (one file spanning the whole period,
rather than AVISO's one-file-per-day pattern).
"""
import numpy as np
import pyinterp.backends.xarray
import xarray as xr
from swot_simulator.plugins import data_handler


class GlorysLoader(data_handler.DatasetLoader):
    def __init__(self, path, var_name='zos', lon_name='lon', lat_name='lat', time_name='time'):
        self.path = path
        self.var_name = var_name
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.time_name = time_name

    def load_dataset(self, first_date: np.datetime64, last_date: np.datetime64) -> xr.Dataset:
        ds = xr.open_dataset(self.path)
        dt = self._calculate_time_delta(ds[self.time_name])
        first_date = self._shift_date(first_date, -1, dt)
        last_date = self._shift_date(last_date, 1, dt)
        ds = ds.sel({self.time_name: slice(first_date, last_date)})
        ds = ds.rename({
            self.lon_name: 'lon', self.lat_name: 'lat',
            self.time_name: 'time', self.var_name: 'ssh',
        })[['lon', 'lat', 'time', 'ssh']]
        # pyinterp identifies geographic axes via CF units attrs, not dim
        # names - assign them defensively (source files should already carry
        # these, but this matches src/utils.py's add_geo_attrs convention).
        ds['lon'] = ds.lon.assign_attrs(units='degrees_east')
        ds['lat'] = ds.lat.assign_attrs(units='degrees_north')
        return ds


class Glorys12(data_handler.CartesianGridHandler):
    """ssh_plugin reading a Glorys (or any lon/lat/time/var NetCDF) source file."""

    def __init__(self, path, var_name='zos'):
        super().__init__(GlorysLoader(path, var_name=var_name))

    def interpolate(self, lon: np.ndarray, lat: np.ndarray, dates: np.ndarray) -> np.ndarray:
        dataset = self.dataset_loader.load_dataset(dates.min(), dates.max())
        interpolator = pyinterp.backends.xarray.Grid3D(dataset.ssh)
        return interpolator.trivariate(
            dict(lon=lon, lat=lat, time=dates),
            interpolator="bilinear",
        )
