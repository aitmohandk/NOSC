from contrib.moving_patches.movpatch_data_tests import MovingPatchDataModuleFastRecGPU, XrDatasetMovingPatchFastRecGPU, XrDatasetMovingPatchFastRecGPUNoFullNaN
from contrib.multivar.multivar_utils import MultivarBatchSelector
import numpy as np
import xarray as xr
import functools as ft
import torch
import time
from dask.diagnostics.progress import ProgressBar
import psutil
import os

#from pathlib import Path
"""
class MultivarDaskXrDataset(MultivarXrDataset):
    def __init__(self, da, domain_limits=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.da = self.collate_fn
        da_dims = dict(zip(self.da.dims, self.da.shape))
        self.da_dims = da_dims

    # 3. Créer une fonction de collation pour charger les batches efficacement
    def collate_fn(self,batch):
        # batch est une liste de Dask Arrays
        stacked = self.da.stack(batch, axis=0)  # Empile les chunks
        return torch.from_numpy(stacked.compute())  # Un seul .compute() pour tout le batch
"""

class _NormalizeFn:
    """Picklable replacement for the local `normalize` closure previously
    defined inside `MultivarDataModule.post_fn()`. A function nested inside
    another function cannot be pickled by the standard `pickle` module
    (no importable qualified name), which breaks `torch.utils.data.DataLoader`
    with `multiprocessing_context='spawn'` (spawned workers need to pickle
    the dataset, including `postpro_fn`). Defining this as a module-level
    class with `__call__` keeps the same behavior while being picklable,
    since only its `m`/`s` numpy arrays need to be pickled.
    """
    def __init__(self, m, s):
        self.m = m
        self.s = s

    def __call__(self, item):
        item_shape_len = len(item.shape)
        m = np.expand_dims(self.m, tuple(range(1, item_shape_len)))
        s = np.expand_dims(self.s, tuple(range(1, item_shape_len)))
        return (item - m) / s


class MultivarXrDataset(XrDatasetMovingPatchFastRecGPU):
    def __init__(self, *args, aug_dims=None, aug_dims_noise=None, aug_dims_offset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_dims = aug_dims
        self.aug_dims_noise = aug_dims_noise
        self.aug_dims_offset = aug_dims_offset
        self._rng = np.random.default_rng()

        if self.aug_dims is not None:
            self.time_perm = np.random.permutation(self.da_dims['time'])
    
    def apply_augmentation(self, item, sl):

        if self.aug_dims is None and self.aug_dims_noise is None and self.aug_dims_offset is None:
            return item
        
        #print("Apply_augmentation")

        if self.aug_dims is not None: 
            for (aug_input_idx, aug_target_idx) in self.aug_dims:
                sl_aug = sl.copy()
                sl_aug['time'] = self.time_perm[sl['time']]
                aug_item = self.da.isel(**sl_aug)
                aug_item_input = np.where(np.isfinite(aug_item.values[aug_input_idx,:]), item.values[aug_target_idx,:], np.full_like(aug_item.values[aug_target_idx,:], np.nan))
                item.values[aug_input_idx,:] = aug_item_input

        if self.aug_dims_noise is not None:
            for (aug_noise_idx, aug_noise_value) in self.aug_dims_noise:
                noise = self._rng.uniform(-aug_noise_value, aug_noise_value, item.values[aug_noise_idx].shape).astype(np.float32)
                item.values[aug_noise_idx] = item.values[aug_noise_idx] + noise

    def get_coords_leadtime(self):
        self.return_coords = True
        coords_sizes = []
        coords_slices = []
        try:
            for i in range(len(self)):
                coords_size, coords_slice = self[i]
                coords_sizes.append(coords_size)
                coords_slices.append(coords_slice)
        finally:
            self.return_coords = False
            return coords_sizes, coords_slices

    def reconstruct_from_items(self, items: torch.Tensor, weight=None, leadtime=0):
        """
            Reconstruction of patches that can contain padded patches
        """

        # getting coords
        start_time = time.time()
        coords_slices = self.get_coords()

        coords_dims = self.patch_dims

        #print("coords_dims")
        #print(coords_dims)

        new_dims = [f'v{i}' for i in range(len(items[0].cpu().shape) - len(coords_dims))]
        dims = new_dims + list(coords_dims)

        new_shape = items[0].shape[:len(new_dims)]
        full_unpadded_shape = [*new_shape, *self.get_unpadded_dims()]
        #full_padded_shape = [*new_shape, *self.get_padded_dims()]

        # create cuda slices
        full_slices = []
        time_cut = items[0].size(dim=1)
        for idx, coord_slices in enumerate(coords_slices):
            coord_slices['time'] = np.arange(coord_slices['time'][leadtime], coord_slices['time'][leadtime] + time_cut)
            full_slices.append(np.ix_(*([np.arange(len_new_dim) for len_new_dim in full_unpadded_shape[:len(new_dims)]]+list(coord_slices.values()))))

        # create cuda tensors
        #rec_tensor = torch.zeros(size=full_padded_shape).cuda()
        #count_tensor = torch.zeros(size=full_padded_shape).cuda()
        from contrib.multivar.device_utils import default_device
        _dev = default_device()
        rec_tensor = torch.zeros(size=full_unpadded_shape, device=_dev)
        count_tensor = torch.zeros(size=full_unpadded_shape, device=_dev)
        w = torch.tensor(weight, device=_dev)

        for idx in range(items.size(0)):
            rec_tensor[full_slices[idx]] += items[idx] * w
            count_tensor[full_slices[idx]] += w
        result_tensor = (rec_tensor / count_tensor).cpu()
        #result_array = np.array(result_tensor[[slice(0,max_shape) for max_shape in full_unpadded_shape]])
        result_array = result_tensor.numpy()

        result_da = xr.DataArray(
            result_array,
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )

        print('total reconstruction time: {:.3f}'.format(time.time() - start_time))
        return result_da

    # ------------------------------------------------------------------
    # Streaming (memory-bounded) reconstruction.
    #
    # `reconstruct_from_items` needs the ENTIRE set of test patches in RAM at
    # once (the caller built one big `torch.cat(self.test_data)`), which for a
    # multi-year test period blows up to tens of GB and OOM-kills the process.
    #
    # The full-domain destination buffers ([n_vars, time, lat, lon]) are tiny
    # compared to the stack of every overlapping patch, and the per-patch
    # `full_slices` depend only on patch POSITION (from get_coords), not on the
    # predicted values. So we can allocate the buffers once and add each batch
    # of patches as it arrives, then drop the patch tensors. This keeps RAM
    # bounded regardless of how long the test period is.
    # ------------------------------------------------------------------
    def init_streaming_rec(self, item_shape, weight, leadtime=0):
        """Allocate zero rec/count buffers and precompute per-patch slices.

        item_shape: shape of a SINGLE patch tensor, e.g. [n_vars, time_cut, H, W]
        (the batch dimension stripped).
        """
        from contrib.multivar.device_utils import default_device

        coords_slices = self.get_coords()
        coords_dims = self.patch_dims

        n_new = len(item_shape) - len(coords_dims)
        new_dims = [f'v{i}' for i in range(n_new)]
        new_shape = list(item_shape[:n_new])
        full_unpadded_shape = [*new_shape, *self.get_unpadded_dims()]

        # first patch_dim is 'time'; time_cut == number of time steps per patch
        time_cut = item_shape[n_new]
        full_slices = []
        for coord_slices in coords_slices:
            coord_slices['time'] = np.arange(
                coord_slices['time'][leadtime],
                coord_slices['time'][leadtime] + time_cut,
            )
            full_slices.append(
                np.ix_(*([np.arange(l) for l in new_shape] + list(coord_slices.values())))
            )

        _dev = default_device()
        self._stream_rec = torch.zeros(size=full_unpadded_shape, device=_dev)
        self._stream_count = torch.zeros(size=full_unpadded_shape, device=_dev)
        self._stream_w = torch.tensor(weight, device=_dev)
        self._stream_slices = full_slices
        self._stream_dims = new_dims + list(coords_dims)

    def add_to_streaming_rec(self, items, offset):
        """Add a batch of patches into the buffers.

        items: tensor [B, *item_shape]; `offset` is the global index of the
        first patch in this batch (== number of patches already processed).
        """
        items = items.to(self._stream_rec.device)
        for i in range(items.size(0)):
            sl = self._stream_slices[offset + i]
            self._stream_rec[sl] += items[i] * self._stream_w
            self._stream_count[sl] += self._stream_w

    def finalize_streaming_rec(self):
        """Return the reconstructed DataArray and release the buffers."""
        result_array = (self._stream_rec / self._stream_count).cpu().numpy()
        result_da = xr.DataArray(
            result_array,
            dims=self._stream_dims,
            coords={d: self.da[d] for d in self.patch_dims},
        )
        self._stream_rec = None
        self._stream_count = None
        self._stream_slices = None
        self._stream_w = None
        return result_da

    def reconstruct_from_items_theo(self, items):
        """
            Reconstruction of patches
        """

        coords_dims = self.patch_dims
        time_crop = self.patch_dims['time']//2

        new_dims = [f'v{i}' for i in range(1)]
        dims = new_dims + list(coords_dims)

        items = np.expand_dims(items, axis=0).astype(np.float32)

        result_da = xr.DataArray(
            items,
            dims=list(dims),
            coords={
                    d: self.da[d][time_crop:-time_crop] if d == 'time' else self.da[d]
                    for d in self.patch_dims
                   },
        )

        print(result_da)


        return result_da

class MultivarDataModule(MovingPatchDataModuleFastRecGPU):

    def __init__(self, multivar_da, domains, xrds_kw, dl_kw, norm_stats=None, aug_dims=None, aug_dims_noise=None, aug_dims_offset=None, **kwargs):
        
        print(f"init datamodule")
        mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # en Go
        print(f"RAM = {mem_used:.2f} Go")

        self.input_da, self.multivar_information = multivar_da
        #print(self.input_da.sel(variable="u_drifter").mean(skipna=True).values.item())
        self.aug_dims = aug_dims
        self.aug_dims_noise = aug_dims_noise
        self.aug_dims_offset=aug_dims_offset
        self._norm_stats = norm_stats
        super().__init__(self.input_da, domains, xrds_kw, dl_kw, norm_stats=norm_stats, **kwargs)
        self.multivar_info()

    # Modified by TP to add norm stat defined option
    def norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats computed", self._norm_stats)
        else:
            self._norm_stats = (np.array(self._norm_stats[0]),np.array(self._norm_stats[1]))
            print("Norm stats defined", self._norm_stats)

        """
        #save norm_stats
        if self.logger:
            print(f"Saving norm at : {Path(self.logger.log_dir)}")
            with open(f"{Path(self.logger.log_dir)}"+'/norm_stats.pkl', 'wb') as f:
                pickle.dump(self._norm_stats, f)
        else:
            print("No self.logger")
        """
        
        return self._norm_stats
    
    def placeholder_norm_stats(self):
        return 0., 1.

    def output_norm_stats(self):
        m,s = self._norm_stats[0][self.multivar_info_dict['full_output_idx']], self._norm_stats[1][self.multivar_info_dict['full_output_idx']]
        return m, s
    
    def input_norm_stats(self):
        return self._norm_stats[0][self.multivar_info_dict['full_input_idx']], self._norm_stats[1][self.multivar_info_dict['full_input_idx']]

    def train_mean_std(self):
        m = []
        s = []

        print("Computing mean and std of training dataset ...")

        #print(self.domains['train'])

        data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])

        mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3  # en Go
        print(f"RAM = {mem_used:.2f} Go")

        for var, var_information in self.multivar_information.items():
            #print(var_information)
            if var.startswith('masked_'):
                m_var, s_var = data.sel(variable=var.split('masked_')[1]).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))
                
            #TP modif for nan
            elif 'fill_nan' in var_information:
                print("Remove fill nan during normalisation")
                data_fill_nan = xr.where(data.sel(variable=var) == var_information["fill_nan"],np.nan,data.sel(variable=var))
                #m_var, s_var = data.sel(variable=var).pipe(lambda da: (da.mean(skipna=True).values.item(), da.std(skipna=True).values.item()))
                m_var, s_var = data_fill_nan.pipe(lambda da: (da.mean(skipna=True).values.item(), da.std(skipna=True).values.item()))
            else:
                m_var, s_var = data.sel(variable=var).pipe(lambda da: (da.mean(skipna=True).values.item(), da.std(skipna=True).values.item()))

            m.append(m_var)
            s.append(s_var)
        return np.array(m), np.array(s)

    def post_fn(self):
        
        m, s = self.norm_stats()
        return _NormalizeFn(m, s)

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only

        post_fn = self.post_fn()
        self.train_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, aug_dims=self.aug_dims,aug_dims_noise=self.aug_dims_noise, aug_dims_offset=self.aug_dims_offset, postpro_fn=post_fn, rand=True
        )
        self.val_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False, aug_dims_offset=self.aug_dims_offset
        )
        self.test_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False, aug_dims_offset=self.aug_dims_offset
        )

    def multivar_info(self):
        full_input_idx = []
        prior_input_idx = []
        full_output_idx = []
        state_obs_channels = []
        state_obs_input_idx = []
        masked_state_obs_input_idx = []
        len_time_channels = self.xrds_kw.patch_dims['time']

        for idx, (var, var_information) in enumerate(self.multivar_information.items()):
            if var_information['input_arch'] == 'full_input':
                full_input_idx.append(idx)
                if var_information['output_arch'] == 'full_output':
                    state_obs_input_idx.append(idx)
                if 'masked_obs' in list(var_information.keys()):
                    state_obs_input_idx.append(idx)
                    # ASSUMES THAT MASKED_OBS IMPLIES THAT ITS UNMASKED VERSION IS A FULL OUTPUT
                    masked_state_obs_input_idx.append(idx+1)
            elif var_information['input_arch'] == 'prior_input':
                prior_input_idx.append(idx)
            if var_information['output_arch'] == 'full_output':
                full_output_idx.append(idx)

        for state_obs_input in state_obs_input_idx:
            if state_obs_input in list(full_output_idx):
                idx = np.argwhere(np.array(full_output_idx) == state_obs_input).item()
                state_obs_channels.extend([idx*len_time_channels+i for i in range(len_time_channels)])

        for masked_state_obs_input in masked_state_obs_input_idx:
            if masked_state_obs_input in list(full_output_idx):
                idx = np.argwhere(np.array(full_output_idx) == masked_state_obs_input).item()
                state_obs_channels.extend([idx*len_time_channels+i for i in range(len_time_channels)])
                
        multivar_info = dict(
            full_input_idx = full_input_idx,
            prior_input_idx = prior_input_idx,
            full_output_idx = full_output_idx,
            state_obs_channels = state_obs_channels,
            state_obs_input_idx = state_obs_input_idx,
            var_names = list(self.multivar_information.keys())
        )
        self.multivar_info_dict = multivar_info

        batch_selector = MultivarBatchSelector()
        batch_selector.multivar_setup(multivar_info)

class MultivarNfNXrDataset(MultivarXrDataset, XrDatasetMovingPatchFastRecGPUNoFullNaN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MultivarNfNDataModule(MultivarDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = MultivarNfNXrDataset(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, aug_dims=self.aug_dims, aug_dims_noise=self.aug_dims_noise, aug_dims_offset=self.aug_dims_offset, postpro_fn=post_fn, rand=True
        )
        self.val_ds = MultivarNfNXrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False, aug_dims_offset=self.aug_dims_offset
        )
        self.test_ds = MultivarNfNXrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False, aug_dims_offset=self.aug_dims_offset
        )
            