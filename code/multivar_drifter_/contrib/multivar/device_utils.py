"""
Central device helper so the multivar code path runs unchanged on GPU or CPU
(laptops, CI, smoke tests). Replaces scattered hardcoded .cuda() calls, which
crash with 'Torch not compiled with CUDA enabled' or 'no CUDA device' on
CPU-only machines.

- default_device(): the process's compute device, overridable via the
  NOSC_DEVICE env var (e.g. NOSC_DEVICE=cpu forces CPU even on a GPU box).
- to_device(x): move a tensor/module to it.
Prefer moving tensors to an existing reference tensor's .device inside model
code (see the test/reconstruction hooks) so everything follows the module's
Lightning-assigned device automatically.
"""
import os
import functools


@functools.lru_cache(maxsize=1)
def default_device():
    import torch
    override = os.environ.get('NOSC_DEVICE')
    if override:
        return torch.device(override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(x, device=None):
    return x.to(default_device() if device is None else device)
