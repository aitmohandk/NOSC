# script.py
#
# Generalizes import_data_glorys_0m_year.py / import_data_glorys_15m_year.py
# (each restricted to a single depth level via minimum_depth == maximum_depth)
# to a single copernicusmarine.subset call spanning a depth *range*, so it
# returns every native Glorys depth level in that range in one file - the
# multi-depth source consumed by open_var_dataset's depth_level selection
# (contrib/data_loading/data.py) for Phase 2 (2D -> 3D) of the multivar
# architecture extension.
import os
import sys

y_start = sys.argv[1]
min_depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.49402499198913574
max_depth = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0

print(f"Year start: {y_start}, depth range: [{min_depth}, {max_depth}]")

# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml qui
# suit la meme convention via ${oc.env:NOSC_DATA_ROOT}. Sous-dossier
# "glorys_raw" = source lue ensuite par prepare_glorys_osse.py.
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "glorys_raw")
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["mlotst", "uo", "vo", "zos", "thetao"],
  minimum_longitude=-180,
  maximum_longitude=179.9166717529297,
  minimum_latitude=-90,
  maximum_latitude=90,
  start_datetime=f"{y_start}-01-01T00:00:00",
  end_datetime=f"{y_start}-12-31T00:00:00",
  minimum_depth=min_depth,
  maximum_depth=max_depth,
  output_directory=folder_data,
)
