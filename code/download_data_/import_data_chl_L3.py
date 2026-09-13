# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "CHL_L3")

import os
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
  variables=["CHL"],
  minimum_longitude=-179.99722290039062,
  maximum_longitude=179.9972381591797,
  minimum_latitude=-89.99722290039062,
  maximum_latitude=89.99722290039062,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2020-01-01T00:00:00",
  output_directory=folder_data
)