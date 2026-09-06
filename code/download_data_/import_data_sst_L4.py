# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "sst_L4")
import os
import copernicusmarine



copernicusmarine.subset(
  dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
  variables=["analysed_sst"],
  minimum_longitude=0,
  maximum_longitude=10,
  minimum_latitude=40,
  maximum_latitude=44,
  start_datetime="2023-05-01T00:00:00",
  end_datetime="2023-09-01T00:00:00",
)


"""
copernicusmarine.subset(
  dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
  variables=["analysed_sst"],
  minimum_longitude=-179.97500610351562,
  maximum_longitude=179.97500610351562,
  minimum_latitude=-89.9749984741211,
  maximum_latitude=89.9749984741211,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2022-01-01T00:00:00",
  output_directory = folder_data,
)
"""