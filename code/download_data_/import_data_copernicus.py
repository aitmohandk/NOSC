# Racine de donnees parametrable (obligatoire) : cf. config/xp/*.yaml
folder_data = os.path.join(os.environ["NOSC_DATA_ROOT"], "uv/GC_daily/daily")

import os
import copernicusmarine


copernicusmarine.subset(
  dataset_id="cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
  variables=["uo", "vo"],
  minimum_longitude=-179.875,
  maximum_longitude=179.875,
  minimum_latitude=-89.875,
  maximum_latitude=89.875,
  start_datetime="2019-02-01T00:00:00",
  end_datetime="2019-02-20T00:00:00",
  minimum_depth=15,
  maximum_depth=15,
  output_directory = folder_data,
)
"""
copernicusmarine.subset(
  dataset_id="cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
  variables=["ue", "ve"],
  minimum_longitude=-179.875,
  maximum_longitude=179.875,
  minimum_latitude=-89.875,
  maximum_latitude=89.875,
  start_datetime="2019-01-01T00:00:00",
  end_datetime="2020-01-01T00:00:00",
  minimum_depth=15,
  maximum_depth=15,
  output_directory = folder_data,
)
"""
"""
copernicusmarine.subset(
  dataset_id="cmems_obs-ins_glo_phy-cur_nrt_drifter_irr",
  dataset_part="history",
  variables=["EWCT", "NSCT"],
  minimum_longitude=-180,
  maximum_longitude=179.99989318847656,
  minimum_latitude=-78.30599975585938,
  maximum_latitude=89.97200012207031,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2019-01-01T00:00:00",
  minimum_depth=0,
  maximum_depth=20,

)
"""