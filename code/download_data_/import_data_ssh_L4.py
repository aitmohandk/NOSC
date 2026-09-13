# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "ssh_L4")
import os
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
  variables=["adt","sla","ugos","vgos"],
  minimum_longitude=-179.9375,
  maximum_longitude=179.9375,
  minimum_latitude=-89.9375,
  maximum_latitude=89.9375,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2024-01-01T00:00:00",
  output_directory = folder_data,
)