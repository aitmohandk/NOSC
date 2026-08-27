from src.models import Lit4dVarNet, GradSolverZero, BilinAEPriorCost, BaseObsCost, ConvLstmGradModel
from contrib.multivar.multivar_models import Multivar4dVarNet
from contrib.multivar.multivar_utils import MultivarBatchSelector
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

#import sys
#sys.path.append("/Odyssey/private/t22picar/4Dvarnet_uv/4dvarnet-starter/contrib/multivar/")
from contrib.multivar.parts import StandardBlock, ResBlock, Down, Up, OutConv

import kornia.filters as kfilts


class MultivarUNet(Multivar4dVarNet):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.premiere_train = True  # Flag pour le premier step
        #print(self.logger)
        #print(self.solver)
        #print(self.trainer.log_dir)
        #self.save_norm_stat()

    def save_norm_stat(self):
        if self.logger:
            print(f"Saving norm at : {Path(self.logger.log_dir)}")
            print(self.norm_stats())
            with open(f"{Path(self.logger.log_dir)}"+'/norm_stats.pkl', 'wb') as f:
                pickle.dump(self._norm_stats, f)
        else:
            print("No self.logger")

    def multivar_step_mask(self, batch, phase=""):

        out = self(batch=batch)
        output_var_names = self.multivar_selector.multivar_output_var_names()
        size_t = out.size(1) // len(output_var_names)

        out = out.view(out.size(0), len(output_var_names), size_t, out.size(2), out.size(3))

        #print(out.shape)
        #print(self.multivar_selector.multivar_full_output(batch).shape)

        loss = None
        total_mse = None

        """
        if phase=="val":
            for i, var in enumerate(output_var_names):
                #TP : add mask nan
                #mask = ~torch.isnan(self.multivar_selector.multivar_obs_input(batch).view_as(out)[:,i])
                mask = (self.multivar_selector.multivar_obs_input(batch).view_as(out)[:,i] != 0).float() # 1 si != de 0 
                loss_i = self.weighted_mse((out[:,i] - self.multivar_selector.multivar_obs_input(batch).view_as(out)[:,i])*mask, self.rec_weight[:out.size(2)])
            #

        else:    
        """ 
        for i, var in enumerate(output_var_names):
            #TP : add mask nan
            #mask = ~torch.isnan(self.multivar_selector.multivar_full_output(batch).view_as(out)[:,i])
            # A changer
            mask = (self.multivar_selector.multivar_full_output(batch).view_as(out)[:,i] != self.multivar_selector.multivar_full_output(batch).view_as(out)[:,i][0][0][0]).float() # 1 si != de 0 
            #print(torch.sum(mask))

            loss_i = self.weighted_mse((out[:,i] - self.multivar_selector.multivar_full_output(batch).view_as(out)[:,i])*mask, self.rec_weight[:out.size(2)])
            with torch.no_grad():
                mse_i = 10000 * loss_i * self.output_norm_stats[1][i]**2
                self.log(f"{phase}_{var}_mse", mse_i, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{phase}_{var}_loss", loss_i, prog_bar=True, on_step=False, on_epoch=True)
            loss = loss_i if loss is None else loss + loss_i
            total_mse = mse_i if total_mse is None else total_mse + mse_i

        with torch.no_grad():
            self.log(f"{phase}_total_mse", total_mse, prog_bar=True, on_step=False, on_epoch=True)
              
        return loss, out

    def step(self, batch, phase=""):

        #if self.premiere_train:
        #    print("C'est le premier validation step !")
        #    self.save_norm_stat()
        #    self.premiere_train=False

        
        # SKIP BATCH TO IMPLEMENT #
        if self.skip_batch(batch):
            return None, None

        #training_loss, out = self.multivar_step(batch, phase)
        training_loss, out = self.multivar_step_mask(batch, phase)

        return training_loss, out
    
    # Add dimensions 
    def forward(self, batch):
        batch_input = self.multivar_selector.multivar_prior_input(batch).nan_to_num()
        return self.solver(batch_input)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, block=ResBlock,
                 add_input=False):
        super(UNet, self).__init__()
        #self.block = ResBlock
        self.n_channels = n_channels
        print("n_channels")
        print(n_channels)
        self.n_classes = n_classes
        print("n_classes")
        print(n_classes)
        self.add_input = add_input
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # block-wise weight scaling factors for stabilised gradients
        sfs = 1/torch.arange(1, 10).sqrt()
        
        # define modules
        self.inc = StandardBlock(n_channels, 64)
        self.down1 = Down(64, 128, block, sf=sfs[1])
        self.down2 = Down(128, 256, block, sf=sfs[2])
        self.down3 = Down(256, 512, block, sf=sfs[3])
        self.down4 = Down(512, 1024 // factor, block, sf=sfs[4])
        
        self.up1 = Up(1024, 512 // factor, block, bilinear, sf=sfs[5])
        self.up2 = Up(512, 256 // factor, block, bilinear, sf=sfs[6])
        self.up3 = Up(256, 128 // factor, block, bilinear, sf=sfs[7])
        self.up4 = Up(128, 64, block, bilinear, sf=sfs[8])
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        if self.add_input:
            inp = x[:,-1].unsqueeze(1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        out = self.outc(x)
        if self.add_input:
            out += inp

        return out

def cosanneal_lr_adam_unet(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.parameters(), "lr": lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def get_multivar_only_prior_dims_in(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.input_arch == 'prior_input':
            dims_in+=1
    return dims_in * channels_per_dim


class MultivarUNet_weight(MultivarUNet):

    def __init__(self,*arg,weight,**kwargs):
        super(self).__init__(*arg,**kwargs)
        self.weight = weight

    def on_test_epoch_end(self):
        self.clear_gpu_mem()
        print('TEST DATA SIZE: {}'.format(torch.cat(self.test_data).size()))

        n_output_dims = self.test_data[0].shape[1]

        for output_dim in range(n_output_dims):
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct_from_items(
                torch.cat(self.test_data).index_select(dim=1, index=torch.Tensor([output_dim]).type(torch.int64)),
                self.weight.cpu().numpy()[:self.weight.cpu().numpy().shape[0]//n_output_dims]
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            metric_data = test_data.pipe(self.pre_metric_fn)
            metrics = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })

            print(metrics.to_frame(name="Metrics").to_markdown())
            if self.logger:
                test_data.to_netcdf(Path(self.logger.log_dir) / f'test_data_dim{output_dim}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_dim{output_dim}.nc')
                self.logger.log_metrics(metrics.to_dict())