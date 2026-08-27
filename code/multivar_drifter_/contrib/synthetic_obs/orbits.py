"""
Closed-form repeat ground-track model for inclined circular orbits.

No SGP4/ephemeris: the ascending-node longitude of orbit n advances by the
mission's exact repeat-geometry drift (shift_per_orbit_deg = 360 x
nodal_days_per_cycle / orbits_per_cycle - see missions.py; this encodes the
J2 nodal precession implicitly and closes the repeat cycle exactly), and the
within-orbit ground track is the standard spherical-triangle
parametrisation of an inclined circular orbit, sheared consistently with
that same per-orbit drift. This reproduces realistic track spacing/shape,
repeat-cycle closure and day-to-day sampling density without needing
precise instantaneous satellite position.
"""
import numpy as np

from contrib.synthetic_obs.missions import Mission

EARTH_RADIUS_KM = 6371.0


def ascending_node_lon_deg(mission: Mission, orbit_number) -> np.ndarray:
    """Ascending-node crossing longitude (deg, unwrapped) of the given orbit number(s)."""
    orbit_number = np.asarray(orbit_number, dtype=np.float64) + mission.phase_offset_orbits
    return mission.ascending_node_lon0_deg - orbit_number * mission.shift_per_orbit_deg


def ground_track(mission: Mission, orbit_number, n_samples=720):
    """
    Ground track (lat, lon) in degrees for one orbit revolution.

    orbit_number: absolute orbit index since the mission's reference ascending node.
    n_samples: number of points sampled along the argument of latitude [0, 360).
    """
    incl = np.radians(mission.inclination_deg)
    u = np.linspace(0.0, 360.0, n_samples, endpoint=False)
    u_rad = np.radians(u)

    lat = np.degrees(np.arcsin(np.sin(incl) * np.sin(u_rad)))
    lon_in_orbit_frame = np.degrees(np.arctan2(np.cos(incl) * np.sin(u_rad), np.cos(u_rad)))

    # within-orbit shear consistent with the per-orbit node drift (so that
    # u=360 lands exactly on the next orbit's ascending node)
    earth_rotation_shear_deg = (mission.shift_per_orbit_deg / 360.0) * u

    lon0 = ascending_node_lon_deg(mission, orbit_number)
    lon = lon0 + lon_in_orbit_frame - earth_rotation_shear_deg
    lon = (lon + 180.0) % 360.0 - 180.0

    return lat, lon


def _bearing_deg(lat, lon):
    """Great-circle bearing (deg) at each point of a track, towards the next point.

    The last point reuses the previous segment's bearing (no next sample to point to).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat1, lon1 = lat_rad[:-1], lon_rad[:-1]
    lat2, lon2 = lat_rad[1:], lon_rad[1:]
    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return np.append(bearing, bearing[-1])


def _destination_point(lat, lon, bearing_deg, distance_km):
    """Spherical destination point given start point, bearing and great-circle distance."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat_rad) * np.cos(delta) + np.cos(lat_rad) * np.sin(delta) * np.cos(bearing_rad)
    )
    lon2 = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(delta) * np.cos(lat_rad),
        np.cos(delta) - np.sin(lat_rad) * np.sin(lat2),
    )
    lat2_deg = np.degrees(lat2)
    lon2_deg = (np.degrees(lon2) + 180.0) % 360.0 - 180.0
    return lat2_deg, lon2_deg


def swath_tracks(mission: Mission, orbit_number, n_samples=720, cross_track_step_km=5.0):
    """
    Points covered by the two wide-swath bands either side of the ground track
    (e.g. SWOT KaRIn), as a list of (lat, lon) point arrays - one per
    cross-track offset sampled, on each side of the nadir gap.

    Returns an empty list if the mission has no wide-swath (swath_width_km == 0).
    """
    if mission.swath_width_km <= 0:
        return []

    lat, lon = ground_track(mission, orbit_number, n_samples=n_samples)
    heading = _bearing_deg(lat, lon)

    half_gap = mission.nadir_gap_km / 2.0
    offsets_km = np.arange(half_gap, half_gap + mission.swath_width_km, cross_track_step_km)
    if offsets_km.size == 0 or offsets_km[-1] < half_gap + mission.swath_width_km:
        offsets_km = np.append(offsets_km, half_gap + mission.swath_width_km)

    tracks = []
    for side_sign in (+1.0, -1.0):
        for offset_km in offsets_km:
            bearing = heading + side_sign * 90.0
            tracks.append(_destination_point(lat, lon, bearing, offset_km))
    return tracks


def orbits_for_day(mission: Mission, day_index: int):
    """Range of absolute orbit numbers whose ground track falls on the given day index."""
    first = int(np.floor(mission.orbits_per_day * day_index))
    last_exclusive = int(np.floor(mission.orbits_per_day * (day_index + 1)))
    return range(first, max(last_exclusive, first + 1))
