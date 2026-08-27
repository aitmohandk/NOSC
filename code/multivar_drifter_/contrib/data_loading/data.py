import xarray as xr
import numpy as np
import pickle
from src.data import TrainingItem
import psutil
import os

def load_ose_data(path):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds.ssh,
            tgt=lambda ds: ds.ssh,
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_ose_data_with_tgt_mask(path, tgt_path, variable='zos'):
    """
        batches need to have a complete target in order for the Grad Masking to be carried out

        path: path to ose data
        tgt_path: path to a complete reconstruction of global glorys ssh containing the day 2020-01-20
        variable: mask variable to load
    """

    ds_mask = xr.open_dataset(tgt_path).drop_vars('depth')
    ds = xr.open_dataset(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})

    ds_mask = ds_mask.isel(time=0)[variable].expand_dims(time=ds.time).assign_coords(ds.coords)

    ds = (
        ds
        .assign(
            input=ds.ssh,
            tgt=ds_mask,
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def mask_input(da, mask_list):
    """Legacy behaviour (kept for open_glorys12_data): applies a RANDOMLY DRAWN
    daily mask to each time step. This destroys the day-to-day progression of
    satellite tracks (repeat-cycle geometry) and is not reproducible across
    runs/workers - prefer mask_input_sequential, which open_var_dataset now uses."""
    i = np.random.randint(0, len(mask_list))
    mask = mask_list[i]
    da = np.where(np.isfinite(mask), da, np.empty_like(da).fill(np.nan)).astype(np.float32)
    return da

def mask_input_sequential(da, mask_array, mode='sequential'):
    """
    Apply mask day t to field day t (deterministic, preserves the temporal
    coherence of the simulated sampling pattern). da: (time, lat, lon) numpy
    array; mask_array: (n_days, lat, lon) with 1.0 observed / NaN elsewhere,
    day 0 of the mask aligned with time step 0 of the dataset (see
    contrib/synthetic_obs/build_masks.py's time_from option, which generates
    the masks from the dataset's own time axis to guarantee this alignment).
    """
    n_time = da.shape[0]
    if mode == 'cycle':
        # deterministic modulo recycling: mask[t % n_masks] -> the intended
        # use of the parent project's ONE-YEAR real L3 mask pickles (365
        # masks over a decade of data), previously served by a random draw.
        # Preserves the intra-year track progression; the year seam is the
        # price of having a single observed year.
        reps = int(np.ceil(n_time / mask_array.shape[0]))
        mask_array = np.concatenate([mask_array] * reps, axis=0)
    elif mask_array.shape[0] < n_time:
        raise ValueError(
            f"mask list has {mask_array.shape[0]} days but the dataset has {n_time} time steps: "
            f"masks are applied sequentially by index and must cover the full period "
            f"(regenerate them with n_days >= {n_time} anchored at the dataset's first date, "
            f"or set mask_mode: cycle to recycle a shorter real-mask set modulo)."
        )
    if mask_array.shape[1:] != da.shape[1:]:
        raise ValueError(
            f"mask spatial shape {mask_array.shape[1:]} does not match data {da.shape[1:]}"
        )
    return np.where(np.isfinite(mask_array[:n_time]), da, np.nan).astype(np.float32)

def open_glorys12_data(path, masks_path, domain, variables="zos", masking=True, test_cut=None):
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    # DROPPING DEPTH !!
    ds =  (
        xr.open_dataset(path).drop_vars('depth')
    )
    
    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    if masking:
        print("OPENING mask list")
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("done.")

        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
    
    ds = ds.sel(domain)
    ds = (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

    return ds

def open_var_dataset(var_path, var, var_name, domain, drop_depth, fill_nan=None, mask_path=None, sst_transfo=None, depth_level=None, depth_index=None):
    """
        open a single dataset for the multivar 4dvar

        var_path: path to load dataset
        var: var name
        domain: domain limits to load
        drop_depth: whether to drop the "depth" var (ignored if depth_level is set)
        mask_path: if not None, loads the .pickle file and masks the input data
        depth_index: if not None, selects this single depth level BY POSITION in the
            file's depth axis (isel) - exact, immune to the nearest-match aliasing
            that depth_level can silently produce when two requested values round
            to the same native level. Preferred for multi-level configs (see
            config/depths/*.yaml + contrib/data_loading/depth_fragments.py).
        depth_level: if not None, selects this single depth level (nearest match) instead
            of dropping the depth dimension entirely - lets one multivar entry represent
            one (variable, depth) pair, e.g. {var_name: thetao, depth_level: 50} for a
            50m-depth temperature entry, following the same "one entry per stacked channel
            group" mechanism already used for distinct variables.
    """
    #print("domain")
    #print(domain)

    # chunks= makes the read dask-backed (lazy): without it, xr.open_dataset
    # returns a NumPy-lazy-indexed array that IS read on slicing, but gets
    # forced fully into memory as soon as it's concatenated across variables
    # (open_multivar_datasets' final .to_array()) - for the full train+val+test
    # time domain and every multivar entry at once, which is what saturates RAM.
    # With dask chunks, that same concatenation stays a lazy graph; only the
    # patches actually consumed downstream get computed.
    raw_dataset = xr.open_dataset(var_path, chunks={'time': 30})
    if 'latitude' in raw_dataset.dims:
        # normalize BEFORE selecting var_name: some multivar entries (e.g.
        # latitude_var) select the coordinate itself as var_name='lat',
        # which only exists post-rename on files using the 'latitude' CF
        # convention (GLORYS) instead of 'lat'.
        raw_dataset = raw_dataset.rename({'latitude': 'lat', 'longitude': 'lon'})
    var_dataset = xr.Dataset({var: raw_dataset[var_name]})

    if 'depth' in var_dataset.dims:
        if depth_index is not None:
            if depth_level is not None:
                raise ValueError(f"var '{var}': set depth_index OR depth_level, not both")
            n_levels = var_dataset.sizes['depth']
            if not (0 <= int(depth_index) < n_levels):
                raise ValueError(f"var '{var}': depth_index {depth_index} out of range [0, {n_levels})")
            var_dataset = var_dataset.isel(depth=int(depth_index))
        elif depth_level is not None:
            var_dataset = var_dataset.sel(depth=depth_level, method='nearest')
        elif drop_depth:
            var_dataset = var_dataset.drop_dims('depth')

    if 'latitude' in list(var_dataset.dims):
        #var_dataset = var_dataset.rename({'latitude':'lat', 'longitude':'lon'})
        var_dataset = var_dataset.rename({'latitude':'lat'})
        
    if 'longitude' in list(var_dataset.dims):
        var_dataset = var_dataset.rename({'longitude':'lon'})

    for domain_var_key in list(domain.keys()):
        if domain_var_key not in var_dataset.dims:
            del domain[domain_var_key]

    var_dataset = var_dataset.sel(domain)
    print(var_dataset[var])

    if sst_transfo is not None:
        print("Changing sst --> log |∇T| ")
        du_dx = (np.abs(var_dataset.differentiate("lon")))
        du_dy = (np.abs(var_dataset.differentiate("lat")))
        epsilone=1e-10
        var_dataset = np.log(du_dx + du_dy + epsilone)
        del du_dx, du_dy, epsilone
        print("Changing sst --> log |∇T| done")

    if fill_nan is not None:
        print("filling nan ...")
        var_dataset = var_dataset.fillna(fill_nan)
        print("filling nan done")

    if mask_path is not None:
        print('masking [{}] var'.format(var))
        mask_var = 'masked_'+var
        with open(mask_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)

        #print("MASKING input data")
        # Sequential, date-aligned masking (mask day t -> field day t). The
        # previous apply_ufunc(mask_input, ...) drew a RANDOM day's mask for
        # each time step, destroying the repeat-cycle track progression and
        # making the input non-reproducible.
        masked_values = mask_input_sequential(var_dataset[var].values, mask_list,
                                              mode=var_info.get('mask_mode', 'sequential'))
        var_dataset = var_dataset.assign({
            mask_var: xr.DataArray(masked_values, dims=var_dataset[var].dims, coords=var_dataset[var].coords)
            })
        #print("MASKING done")
        var_dataset = xr.Dataset({mask_var: var_dataset[mask_var]})

        return var_dataset
    
    return var_dataset

def merge_datasets(original_dataset: xr.Dataset, new_dataset: xr.Dataset, broadcast_time=False):
    if original_dataset is None:
            return new_dataset

    # FIRST VAR CANNOT BE TIME BROADCASTED
    if broadcast_time:
        time_coords = original_dataset.coords['time']

        new_dataset = new_dataset.reindex({'lat': original_dataset.lat, 'lon': original_dataset.lon}, method='nearest')
        new_dataset = new_dataset.expand_dims({'time': time_coords}, axis=0).broadcast_like(original_dataset)

    # TP : I think .assign interpolate if not the same grid ?! Can lead to issues be carefull
    merged_dataset = original_dataset.assign({var_name:var_data for var_name, var_data in new_dataset.data_vars.items()})

    return merged_dataset

# general function to load multiple varaibles from multiple datasets into 4DVarNet
def open_multivar_datasets(vars_info,
                           domain,
                           full_time_domain,
                           drop_depth=True):

    #print(domain)
    # works only if train, val and test slices are in chronological order
    #Modified by TP in case val is later than test
    
    print(f"init open_multivar_datasets")
    mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # en Go
    print(f"RAM = {mem_used:.2f} Go")

    if full_time_domain['val']['time'].stop > full_time_domain['test']['time'].stop:
        domain['time'] = slice(full_time_domain['train']['time'].start, full_time_domain['val']['time'].stop)
    else: 
        domain['time'] = slice(full_time_domain['train']['time'].start, full_time_domain['test']['time'].stop)
    print(domain)

    input_variables = []
    tgt_variables = []
    multivar_information = dict()

    full_dataset = None

    for var, var_info in vars_info.items():
        print('\nhandling [{}] var'.format(var))

        var_path = var_info['var_path']
        var_mask_path = None
        if 'mask_path' in var_info:
            var_mask_path = var_info['mask_path']
        broadcast_time = var_info['broadcast_time']
        fill_nan = None
        sst_transfo=None
        depth_level = var_info.get('depth_level', None)
        depth_index = var_info.get('depth_index', None)
        # var_info_dict
        var_information_dict = dict()
        var_information_dict['input_arch'] = var_info.input_arch
        var_information_dict['output_arch'] = var_info.output_arch

        if 'fill_nan' in var_info:
            #print('fill_nan')
            fill_nan = var_info['fill_nan']
            var_information_dict['fill_nan']=var_info['fill_nan']

        if 'sst_transfo' in var_info:
            #print('fill_nan')
            sst_transfo = True

        if var_mask_path is not None:
            var_dataset = open_var_dataset(var_path, var, var_info.var_name, domain, drop_depth, fill_nan=fill_nan, mask_path=var_mask_path, depth_level=depth_level, depth_index=depth_index)
            full_dataset = merge_datasets(full_dataset, var_dataset)
            var_information_dict_masked = var_information_dict.copy()
            var_information_dict_masked['output_arch'] = 'no_output'
            var_information_dict_masked['masked_obs'] = True
            multivar_information['masked_'+var] = var_information_dict_masked

            var_information_dict['input_arch'] = 'no_input'

        print(f"for var open_multivar_datasets")
        mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # en Go
        print(f"RAM = {mem_used:.2f} Go")

        var_dataset = open_var_dataset(var_path, var, var_info.var_name, domain, drop_depth, fill_nan=fill_nan, sst_transfo=sst_transfo, depth_level=depth_level, depth_index=depth_index)
        full_dataset = merge_datasets(full_dataset, var_dataset, broadcast_time=broadcast_time)
        multivar_information[var] = var_information_dict
        del var_dataset

    full_dataset = (
        full_dataset
        .sel(domain)
        [list(multivar_information.keys())]
        .transpose("time", "lat", "lon", ...)
        .to_array()
    )

    #print(full_dataset)
    taille_go = full_dataset.nbytes / (1024 ** 3)
    print(f"Taille du Dataset : {taille_go:.6f} Go")
    
    print(full_dataset.var())

    print(f"end open_multivar_datasets")
    mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # en Go
    print(f"RAM = {mem_used:.2f} Go")

    return full_dataset, multivar_information