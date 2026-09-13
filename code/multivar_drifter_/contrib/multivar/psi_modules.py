import os

import numpy as np
import torch as torch
import xarray as xr

#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
#: Degrees / radians conversion factor
P0 = torch.pi / 180

def sphere_distance(_lats, _late, _lons, _lone):
    dlat, dlon = P0 * (_late - _lats), P0 * (_lone - _lons)
    return EARTH_RADIUS * torch.sqrt(dlat ** 2 + torch.cos(P0 * _lats) * torch.cos(P0 * _late) * dlon ** 2)

def compute_coriolis_factor(lat):
    return 2 * EARTH_ANG_SPEED * torch.sin(lat * P0)

# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml.
# NB : module non reference par aucune config a ce jour ; corrige par coherence.
file_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "glorys_15m", "glorys_multivar_15m_2010-2018.nc")
maps_4th = xr.open_dataset(file_data).sel(time="2010-01-01")
lon_4th = torch.tensor(maps_4th.lon.values)
lat_4th = torch.tensor(maps_4th.lat.values)
lat_4th, lon_4th = torch.meshgrid(lat_4th, lon_4th)

coriolis_factor_t = compute_coriolis_factor(lat_4th)

dx, dy = torch.zeros_like(lat_4th), torch.zeros_like(lat_4th)
dx[:, :-1] = sphere_distance(lat_4th[:, :-1], lat_4th[:, 1:], lon_4th[:, :-1], lon_4th[:, 1:])
dx[:, -1] = dx[:, -2]

dy[:-1, :] = sphere_distance(lat_4th[:-1, :], lat_4th[1:, :], lon_4th[:-1, :], lon_4th[1:, :])
dy[-1, :] = dy[-2, :]
