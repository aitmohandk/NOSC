# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "sst_L3")
import os
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-sst_glo_phy_my_l3s_P1D-m",
  variables=["adjusted_sea_surface_temperature"],
  minimum_longitude=-179.9499969482422,
  maximum_longitude=179.9499969482422,
  minimum_latitude=-90,
  maximum_latitude=90,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2020-01-01T00:00:00",
  output_directory = folder_data
)
