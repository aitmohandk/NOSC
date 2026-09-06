# script.py
import os
import sys

# Récupérer les arguments passés au script
y_start = sys.argv[1]

print(f"Year start: {y_start}")

# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "glorys_15m")
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
  minimum_depth=15,
  maximum_depth=15,
  output_directory = folder_data
)
