from src.models import Lit4dVarNet, GradSolverZero, BilinAEPriorCost, BaseObsCost, ConvLstmGradModel
from contrib.multivar.multivar_models import Multivar4dVarNet
from contrib.multivar.multivar_utils import MultivarBatchSelector
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append("/Odyssey/private/t22picar/4Dvarnet_uv/4dvarnet-starter/contrib/multivar/")
from contrib.multivar.parts import StandardBlock, ResBlock, Down, Up, OutConv
import kornia.filters as kfilts

from contrib.multivar.psi_modules import coriolis_factor_t,dx,dy,GRAVITY

class MultivarUNet_psi(Multivar4dVarNet):
    def __init__(self, multivar_selector, *args, **kwargs):
        # registered as buffers so they follow the module's device (CPU or GPU)
        # via Lightning's .to(device); previously pinned to "cuda", crashing on
        # CPU-only machines.
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)
        self.register_buffer("coriolis_factor_t", coriolis_factor_t)
        self.GRAVITY = GRAVITY

        super().__init__(multivar_selector, *args, **kwargs)

    def multivar_step_psi(self, batch, phase=""):
        out = self(batch=batch)
        output_var_names = self.multivar_selector.multivar_output_var_names()
        size_t = out.size(1) // len(output_var_names)

        if len(output_var_names)!=2:
            print("DIM output_var_names =! 2")
            pass

        out = out.view(out.size(0), len(output_var_names), size_t, out.size(2), out.size(3))


        """
        
        COMPUTE UTOT,VTOT FROM POTENTIAL PSI, PSI_HAT

        """
        
        #print("PSI CONFIG")
        psi = out[:,0]
        psi_hat = out[:,1]

        dx = self.dx.unsqueeze(0).repeat(psi.size(0), psi.size(1), 1, 1)
        dy = self.dy.unsqueeze(0).repeat(psi.size(0), psi.size(1), 1, 1)
        coriolis_factor_t = self.coriolis_factor_t.unsqueeze(0).repeat(psi.size(0), psi.size(1), 1, 1)

        ## ∇⊥psi : corresponds to the usual geostrophic velocity
        ## ∇⊥psi = k×∇Ψ (cross product with Id) = (-d/dy psi, d/dx psi) x (f(lat)/g) --> Utiliser code pour vitesse geos

        #print(dx.device)
        dpsi_dx, dpsi_dy = torch.zeros_like(psi), torch.zeros_like(psi)  # inherit psi's device
        dpsi_dx[:,:,:,1:-1] = psi[:,:,:,2:] - psi[:,:,:,:-2]
        dpsi_dy[:,:,1:-1,:] = psi[:,:,2:,:] - psi[:,:,:-2,:]

        vgeo = -self.GRAVITY * ( dpsi_dx / dx ) / coriolis_factor_t
        ugeo = self.GRAVITY * ( dpsi_dy / dy ) / coriolis_factor_t

        ## ∇psi_hat = (d/dx psi_hat, d/dy psi_hat) --> Même gradient que fct geo() (en pytorch)
        dpsi_hat_dx, dpsi_hat_dy = torch.zeros_like(psi_hat), torch.zeros_like(psi_hat)

        dpsi_hat_dx[:,:,:,1:-1] = psi_hat[:,:,:,2:] - psi_hat[:,:,:,:-2]
        dpsi_hat_dy[:,:,1:-1,:] = psi_hat[:,:,2:,:] - psi_hat[:,:,:-2,:]

        uageo = dpsi_hat_dx / dx
        vageo = dpsi_hat_dy / dy

        utot = ugeo + uageo 
        vtot = vgeo + vageo 

        out[:,0] = utot
        out[:,1] = vtot

        loss = None
        total_mse = None

        for i, var in enumerate(output_var_names):
            loss_i = self.weighted_mse(out[:,i] - self.multivar_selector.multivar_full_output(batch).view_as(out)[:,i], self.rec_weight[:out.size(2)])
            with torch.no_grad():
                mse_i = 10000 * loss_i * self.output_norm_stats[1][i]**2
                self.log(f"{phase}_{var}_mse", mse_i, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{phase}_{var}_loss", loss_i, prog_bar=True, on_step=False, on_epoch=True)
            loss = loss_i if loss is None else loss + loss_i
            total_mse = mse_i if total_mse is None else total_mse + mse_i

        with torch.no_grad():
            self.log(f"{phase}_total_mse", total_mse, prog_bar=True, on_step=False, on_epoch=True)
              
        out[:,0] = psi
        out[:,1] = psi_hat

        return loss, out

    def step(self, batch, phase=""):

        # SKIP BATCH TO IMPLEMENT #
        if self.skip_batch(batch):
            return None, None

        #training_loss, out = self.multivar_step(batch, phase)
        training_loss, out = self.multivar_step_psi(batch, phase)

        return training_loss, out
    
    # Add dimensions 
    def forward(self, batch):

        batch_input = self.multivar_selector.multivar_prior_input(batch).nan_to_num()
        return self.solver(batch_input)