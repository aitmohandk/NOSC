import numpy as np
from src.utils import get_constant_crop
import torch
# default_device import removed: the selector now aligns index tensors to the
# incoming batch device (see _idx_on) rather than a device fixed at build time.

def get_multivar_aug_dims_noise(multivar_dict):
    aug_dims_noise = None
    dim = 0

    for var, var_info in multivar_dict.items():
        if 'aug_noise' in var_info.keys():
            if var_info.aug_noise:
                if aug_dims_noise is None:
                    aug_dims_noise = []
    # THE AUGMENTATION WILL ALWAYS WORK ON THE NEXT DIMENSION ONLY
                aug_dims_noise.append((dim,var_info['aug_noise']))

        if 'mask_path' in var_info.keys():
            dim += 1
        dim+=1

    return aug_dims_noise

'''
def get_multivar_aug_dims_offset(multivar_dict):
    aug_dims_offset = None
    dim = 0

    for var, var_info in multivar_dict.items():
        if 'aug_offset' in var_info.keys():
            print("aug_offset in var info")
            if var_info.aug_offset:
                if aug_dims_offset is None:
                    aug_dims_offset = []
    # THE AUGMENTATION WILL ALWAYS WORK ON THE NEXT DIMENSION ONLY
                aug_dims_offset.append((dim,var_info['aug_offset']))
                print(f"offset {var}")

        if 'mask_path' in var_info.keys():
            dim += 1
        dim+=1

    return aug_dims_offset
'''

def get_multivar_aug_dims(multivar_dict):
    aug_dims = None
    dim = 0

    for var, var_info in multivar_dict.items():
        if 'aug' in var_info.keys():
            if var_info.aug:
                if aug_dims is None:
                    aug_dims = []
    # THE AUGMENTATION WILL ALWAYS WORK ON THE NEXT DIMENSION ONLY
                aug_dims.append((dim, dim+1))

        if 'mask_path' in var_info.keys():
            dim += 1
        dim+=1

    return aug_dims

def get_multivar_prior_dims_in(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.input_arch == 'prior_input' or var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_prior_dims_out(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_output_var_count(multivar_dict):
    """Number of full_output entries (variables, not channels) - used to size one
    learnable weight per output variable, e.g. for uncertainty-weighted loss combination."""
    return sum(1 for var_info in multivar_dict.values() if var_info.output_arch == 'full_output')


def get_multivar_head_groups(multivar_dict, channels_per_dim, default_group=None):
    """
    Group full_output entries into named channel blocks for a per-group
    "head" architecture (see contrib/multivar/multivar_models_unet_heads.py).

    Each entry's `head_group` config key (default: the entry's own name, i.e.
    its own dedicated head) assigns it to a group; entries sharing the same
    head_group are given ONE shared head and must be contiguous in the
    multivar dict's full_output ordering, since that ordering is exactly the
    channel layout MultivarBatchSelector produces/expects
    (contrib/multivar/multivar_utils.py's multivar_full_output / the
    per-variable loss loop in multivar_step_mask) - raises if violated.

    Returns an OrderedDict {group_name: n_channels}, in output channel order.
    """
    groups = {}
    last_group = None
    for var, var_info in multivar_dict.items():
        if var_info.output_arch != 'full_output':
            continue
        group = var_info.get('head_group', default_group) or var
        if group in groups and group != last_group:
            raise ValueError(
                f"head_group '{group}' is not contiguous in the multivar dict's full_output "
                f"ordering (variables in the same head_group must be listed next to each other)"
            )
        groups[group] = groups.get(group, 0) + channels_per_dim
        last_group = group
    return groups


def get_multivar_grad_dims(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_triang_time_wei(patch_dims, dims_out, offset=0, **crop_kw):
    patch_dims = patch_dims.copy()
    patch_dims["time"] = dims_out
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_multivar_forecast_wei(patch_dims, dims_out, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used

    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_dims['time'] - 1) // 2),
         np.linspace(1, 0.5, 7),
         np.zeros((patch_dims['time'] + 1) // 2 - 7)),
        axis=0)
    
    # assuming dims_out = time * n_vars_out
    n_vars_out = dims_out // patch_dims['time']
    time_patch_weight = np.concatenate([time_patch_weight]*n_vars_out, axis=0)
    pw = np.concatenate([pw]*n_vars_out, axis=0)

    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight

def get_multivar_mapping_wei(patch_dims, dims_out, offset=0, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used

    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )
    
    # assuming dims_out = time * n_vars_out
    n_vars_out = dims_out // patch_dims['time']
    final_patch_weight = np.concatenate([time_patch_weight]*n_vars_out, axis=0)

    return final_patch_weight

def get_multivar_mapping_wei_dirac(patch_dims, dims_out, offset=0, **crop_kw):
    """
    return a dirac weighted fonction at the center of the patch
    """
    pw = get_constant_crop(patch_dims, **crop_kw)

    time_patch_weight = np.fromfunction(
        lambda t, *a: (((offset + 2 * t - patch_dims["time"]))),
        patch_dims.values(),
    )

    for t in range(patch_dims.time):
        time_patch_weight[t,:,:] = np.where(time_patch_weight[t,:,:]==0,pw[t],0)
    
    # assuming dims_out = time * n_vars_out
    n_vars_out = dims_out // patch_dims['time']
    final_patch_weight = np.concatenate([time_patch_weight]*n_vars_out, axis=0)

    return final_patch_weight

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class MultivarBatchSelector(metaclass=SingletonMeta):

    def __init__(self):
        super().__init__()

    def mask_batch(self, batch):
        masked_batch = batch
        masked_dim_size = masked_batch.shape[2]
        masked_batch[:,self.full_input_idx, masked_dim_size//2:] = np.nan
        masked_batch[:,self.prior_input_idx, masked_dim_size//2:] = np.nan

        return masked_batch

    def multivar_setup(self, multivar_info):

        # Kept on CPU at construction; each index_select call aligns the index
        # to the incoming batch's device (see _idx_on). This makes the selector
        # correct on CPU, single-GPU and multi-GPU (DDP) alike, instead of
        # pinning to a single device guessed at build time.
        self.full_input_idx = torch.Tensor(multivar_info['full_input_idx']).type(torch.int64)
        self.prior_input_idx = torch.Tensor(multivar_info['prior_input_idx']).type(torch.int64)
        self.full_output_idx = torch.Tensor(multivar_info['full_output_idx']).type(torch.int64)
        self.state_obs_channels = torch.Tensor(multivar_info['state_obs_channels']).type(torch.int64)
        self.state_obs_input_idx = torch.Tensor(multivar_info['state_obs_input_idx']).type(torch.int64)
        self.var_names = multivar_info['var_names']
        self.output_var_names = [self.var_names[idx] for idx in self.full_output_idx.tolist()]

        print('full_input_idx: {}'.format(list(self.full_input_idx)))
        print('prior_input_idx: {}'.format(list(self.prior_input_idx)))
        print('full_output_idx: {}'.format(list(self.full_output_idx)))
        print('state_obs_channels: {}'.format(list(self.state_obs_channels)))
        print('state_obs_input_idx: {}'.format(list(self.state_obs_input_idx)))

    @staticmethod
    def _idx_on(index, ref):
        """Return `index` on the same device as tensor `ref` (no-op if already there)."""
        return index if index.device == ref.device else index.to(ref.device)

    def multivar_full_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self._idx_on(self.full_input_idx, batch))
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_full_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)

    def multivar_prior_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self._idx_on(self.prior_input_idx, batch))
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_prior_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_full_output(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self._idx_on(self.full_output_idx, batch))
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('new batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_state_obs(self, state):
        new_batch = torch.index_select(state, dim=1, index=self._idx_on(self.state_obs_channels, state))
        #print('multivar_state_obs batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_obs_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self._idx_on(self.state_obs_input_idx, batch))
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_obs_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_output_var_names(self):
        return self.output_var_names