"""
Regional, parameterized GLORYS12 download for the OSSE configs
(osse3d_gs21_*), replacing the global hardcoded-path scripts (which pull
~1.5 TB/year at 1/12 deg - unusable outside the original cluster).

Requires prior authentication:  copernicusmarine login
Requires copernicusmarine >= 2.x (the parent project's env pinned 1.3.3, which
the Copernicus Marine backend now rejects with "Client version is not
compatible" - upgrade with `pip install -U copernicusmarine` /
`conda update copernicusmarine` if you hit that error).

Modes:
  surface    : zos only, top level              (~30 MB/year on the GS box)
  multidepth : thetao,uo,vo, depth in [0,200m]  (~2.5 GB/year on the GS box)
  bathy      : static deptho                     (one-off, ~1 MB)
  merge      : concatenate the yearly files of a directory into one NetCDF

Default domain: Gulf Stream box of config/xp/osse3d_gs21_*.yaml (32-44N,
66-54W) + 0.5 deg margin.

Example (full preparation for the decade):
  python import_data_glorys_region.py --mode surface    --start-year 2010 --end-year 2019 --out <root>/glorys_surface_yearly
  python import_data_glorys_region.py --mode multidepth --start-year 2010 --end-year 2019 --out <root>/glorys_multidepth_yearly
  python import_data_glorys_region.py --mode bathy      --out <root>
  python import_data_glorys_region.py --mode merge --src <root>/glorys_surface_yearly    --out <root>/glorys_gs_surface_2010-2020.nc
  python import_data_glorys_region.py --mode merge --src <root>/glorys_multidepth_yearly --out <root>/glorys_gs_multidepth_2010-2020.nc
"""
import argparse
from pathlib import Path

DATASET_DAILY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_STATIC = "cmems_mod_glo_phy_my_0.083deg_static"


def _subset(out_dir, variables, year=None, args=None, depth=None, output_filename=None):
    import copernicusmarine
    kwargs = dict(
        dataset_id=DATASET_DAILY if year else DATASET_STATIC,
        variables=variables,
        minimum_longitude=args.lon_min, maximum_longitude=args.lon_max,
        minimum_latitude=args.lat_min, maximum_latitude=args.lat_max,
        output_directory=str(out_dir),
        # explicit filenames make yearly calls idempotent instead of
        # accumulating "_(1).nc", "_(2).nc" siblings on re-runs
        output_filename=output_filename,
        skip_existing=True,
    )
    if year:
        kwargs.update(start_datetime=f"{year}-01-01T00:00:00",
                      end_datetime=f"{year}-12-31T00:00:00")
    if depth:
        kwargs.update(minimum_depth=depth[0], maximum_depth=depth[1])
    response = copernicusmarine.subset(**kwargs)
    print(response)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=['surface', 'multidepth', 'bathy', 'merge'])
    p.add_argument('--start-year', type=int, default=2010)
    p.add_argument('--end-year', type=int, default=2019, help='inclusive')
    p.add_argument('--lat-min', type=float, default=31.5)
    p.add_argument('--lat-max', type=float, default=44.5)
    p.add_argument('--lon-min', type=float, default=-66.5)
    p.add_argument('--lon-max', type=float, default=-53.5)
    p.add_argument('--depth-max', type=float, default=200.0)
    p.add_argument('--src', default=None, help='(merge) directory of yearly .nc files')
    p.add_argument('--out', required=True, help='output directory (surface/multidepth/bathy) or output file (merge)')
    args = p.parse_args()

    if args.mode == 'merge':
        # NB: xr.open_mfdataset(...).to_netcdf(...) is NOT lazy enough here:
        # with 10 yearly files of ~3 GB (uncompressed) each, xarray/dask ends
        # up materializing far more than one file's worth of data at once,
        # which saturates RAM. Stream the merge instead, one file (and one
        # variable) at a time, appending along the unlimited 'time' dimension
        # so peak memory stays bounded to a single variable/year (~1 GB).
        import netCDF4 as nc
        files = sorted(Path(args.src).glob('*.nc'))
        if not files:
            raise SystemExit(f"no .nc files in {args.src}")
        print(f"merging {len(files)} files -> {args.out}")

        out_path = Path(args.out)
        if out_path.exists():
            out_path.unlink()

        with nc.Dataset(files[0]) as src0, nc.Dataset(out_path, 'w', format='NETCDF4') as dst:
            dst.setncatts({k: src0.getncattr(k) for k in src0.ncattrs()})
            for name, dim in src0.dimensions.items():
                dst.createDimension(name, None if name == 'time' else len(dim))
            for name, var in src0.variables.items():
                fill_value = getattr(var, '_FillValue', None)
                dst_var = dst.createVariable(name, var.dtype, var.dimensions, fill_value=fill_value)
                dst_var.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != '_FillValue'})

        time_size = 0
        with nc.Dataset(out_path, 'a') as dst:
            for f in files:
                print(f"  + {f.name}")
                with nc.Dataset(f) as src:
                    n = src.dimensions['time'].size
                    for name, var in src.variables.items():
                        if 'time' in var.dimensions:
                            idx = [slice(None)] * len(var.dimensions)
                            idx[var.dimensions.index('time')] = slice(time_size, time_size + n)
                            dst.variables[name][tuple(idx)] = var[:]
                        else:
                            dst.variables[name][:] = var[:]
                time_size += n

        print("done, total time steps:", time_size)
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'bathy':
        _subset(out, ["deptho"], args=args, output_filename="bathymetry.nc")
    elif args.mode == 'surface':
        for year in range(args.start_year, args.end_year + 1):
            print(f"[surface] {year}")
            _subset(out, ["zos"], year=year, args=args, depth=(0.4, 0.6),
                    output_filename=f"glorys_surface_{year}.nc")
    elif args.mode == 'multidepth':
        for year in range(args.start_year, args.end_year + 1):
            print(f"[multidepth] {year} (0-{args.depth_max} m)")
            _subset(out, ["thetao", "uo", "vo"], year=year, args=args, depth=(0.0, args.depth_max),
                    output_filename=f"glorys_multidepth_{year}.nc")


if __name__ == '__main__':
    main()
