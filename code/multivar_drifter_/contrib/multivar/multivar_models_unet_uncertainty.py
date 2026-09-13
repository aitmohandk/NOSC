"""
Learned per-variable loss weighting (homoscedastic uncertainty weighting,
Kendall, Gal & Cipolla 2018 "Multi-Task Learning Using Uncertainty to Weigh
Losses"), as an alternative to MultivarUNet_mae's plain unweighted sum of
per-variable losses - see the conversation this module was written for.

weighted_mae (src/models.py-derived) is an L1 loss, so the Laplace-noise
form of the same idea is used: for scale b_i = exp(s_i), the negative
log-likelihood of an L1 residual is |x|/b_i + log(b_i); reparametrised in
s_i this is exp(-s_i) * L_i + s_i - same functional form as Kendall's
Gaussian derivation, just with the noise model matching the L1 loss already
in use. s_i is a learned scalar per output variable, initialised to 0
(b_i = 1, i.e. no reweighting at the start of training).
"""
import torch
import torch.nn as nn

from contrib.multivar.multivar_models_unet_mae import MultivarUNet_mae
from contrib.multivar.multivar_utils import get_multivar_output_var_count


class MultivarUNet_mae_UncertaintyWeighted(MultivarUNet_mae):
    def __init__(self, n_output_vars, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_vars = nn.Parameter(torch.zeros(n_output_vars))

    def combine_losses(self, per_var_losses, output_var_names, phase=""):
        losses = torch.stack(per_var_losses)
        precision = torch.exp(-self.log_vars)
        weighted = precision * losses + self.log_vars

        with torch.no_grad():
            for i, var in enumerate(output_var_names):
                self.log(f"{phase}_{var}_log_var", self.log_vars[i], on_step=False, on_epoch=True)
                self.log(f"{phase}_{var}_weighted_loss", weighted[i], on_step=False, on_epoch=True)

        return weighted.sum()


def cosanneal_lr_adam_unet_uncertainty(lit_mod, lr, T_max=100, weight_decay=0., log_var_lr=None):
    """Same as contrib.multivar.multivar_models_unet.cosanneal_lr_adam_unet, plus
    the learned per-variable log-variance weights as a second parameter group
    (opt_fn only optimizes lit_mod.solver.parameters() by default, which would
    silently skip lit_mod.log_vars)."""
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.parameters(), "lr": lr},
            {"params": [lit_mod.log_vars], "lr": log_var_lr if log_var_lr is not None else lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }
