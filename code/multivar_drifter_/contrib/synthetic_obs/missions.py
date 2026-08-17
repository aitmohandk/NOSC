"""
Registry of real altimetry missions, parameterized by the quantities that
actually drive ground-track sampling geometry: inclination, repeat-cycle
length and the number of orbits per repeat cycle. These are published,
near-constant mission-design parameters (not live ephemeris), consistent with
a closed-form repeat-orbit ground-track model (see orbits.py) rather than a
full SGP4/TLE propagator.

Numbers are representative literature values (mission handbooks / CNES-AVISO
documentation), not exact osculating-element fits - sufficient for
reproducing realistic sampling density/gap statistics, which is what this
module is for.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Mission:
    name: str
    inclination_deg: float
    repeat_cycle_days: float
    orbits_per_cycle: int
    swath_width_km: float = 0.0
    nadir_gap_km: float = 0.0
    ascending_node_lon0_deg: float = 0.0
    phase_offset_orbits: float = 0.0

    @property
    def orbits_per_day(self) -> float:
        return self.orbits_per_cycle / self.repeat_cycle_days

    @property
    def orbital_period_s(self) -> float:
        return 86400.0 * self.repeat_cycle_days / self.orbits_per_cycle


MISSIONS = {
    # Jason-3 / TOPEX-Poseidon / Sentinel-6 Michael Freilich reference orbit
    "jason3": Mission("jason3", inclination_deg=66.04, repeat_cycle_days=9.9156, orbits_per_cycle=254),
    "sentinel6": Mission("sentinel6", inclination_deg=66.04, repeat_cycle_days=9.9156, orbits_per_cycle=254,
                          phase_offset_orbits=127),  # interleaved half-cycle offset vs jason3
    # Sentinel-3A/B reference orbit
    "sentinel3a": Mission("sentinel3a", inclination_deg=98.65, repeat_cycle_days=27.0, orbits_per_cycle=385),
    "sentinel3b": Mission("sentinel3b", inclination_deg=98.65, repeat_cycle_days=27.0, orbits_per_cycle=385,
                           phase_offset_orbits=192),  # interleaved half-cycle offset vs sentinel3a
    # SARAL/AltiKa (ex-Envisat orbit)
    "saral": Mission("saral", inclination_deg=98.55, repeat_cycle_days=35.0, orbits_per_cycle=501),
    # HY-2B
    "hy2b": Mission("hy2b", inclination_deg=99.34, repeat_cycle_days=14.0, orbits_per_cycle=228),
    # SWOT KaRIn: wide-swath, two 50 km swaths either side of a 20 km nadir gap
    "swot": Mission("swot", inclination_deg=77.6, repeat_cycle_days=20.8646, orbits_per_cycle=292,
                     swath_width_km=50.0, nadir_gap_km=20.0),
}

# Nadir-only constellation resembling the "6 sats" merged product referenced
# elsewhere in this repo (process_data/mask/glorys_masking.ipynb).
SIX_SAT_NADIR = ["jason3", "sentinel6", "sentinel3a", "sentinel3b", "saral", "hy2b"]
