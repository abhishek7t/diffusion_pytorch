"""
CelebaHQ 256x256
"""

import functools
import fire
import numpy as np
import torch
from torch import nn
import os

from diffusion_pt.diffusion_utils import GaussianDiffusion, get_beta_schedule
from diffusion_pt.models import unet
from diffusion_pt.gpu_utils import datasets

class Model(nn.Module):
    def __init__(self, *, model_name, betas: np.ndarray, loss_type: str, num_classes: int,
                dropout: float, randflip, block_size: int):
        self.model_name = model_name
        self.diffusion = GaussianDiffusion(betas=betas, loss_type=loss_type)
        self.num_classes = num_classes
        self.dropout = dropout
        self.randflip = randflip
        self.block_size = block_size
        
        self.unet = unet.UNet2DModel(num_classes=num_classes,
                               ch=128, ch_mult=(1, 1, 2, 2, 4, 4),
                               num_res_blocks=2, attn_resolutions=(16,),
                               out_ch=3, dropout=dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, dropout):
        B, C, H, W = x.shape
        assert x.dtype == torch.float32
        assert t.shape == (B,) # timesteps should have shape [B]
        assert y.shape == (B,)
        
        if self.block_size != 1:
            x = nn.functional.unfold(x, kernel_size=block_size)
        
        # pass the input through the unet model
        if self.model_name == 'unet2d16b2c112244':  # 114M for block_size=1
            out = self.unet(x, t, y)
        else:
            NotImplementedError(self.model_name)
        
        if self.block_size != 1:
            out = nn.functional.fold(out, output_size=(H, W), kernel_size=self.block_size)
        
        assert out.shape == (B, C, H, W)
        return out
    
  
data_dir = '../datasets/celebahq256/celeba_hq_256/'
torch.manual_seed(5)

dataset = 'celebahq256'
optimizer = 'adam'
total_bs = 64
grad_clip = 1
lr = 0.00002
warmup = 5000
num_diffusion_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
beta_schedule = 'linear'
loss_type = 'noisepred'
dropout = 0.0
randflip = 1
block_size = 1,
log_dir = 'logs'

ds = datasets.get_dataloader(name=dataset, data_dir=data_dir, batch_size=8, shuffle=True)

model_dir = log_dir + '/models/celebhq_diffusion'
os.makedirs(model_dir, exist_ok=True)

model_name='unet2d16b2c112244'

model = Model(
        model_name=model_name,
        betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
        ),
        loss_type=loss_type,
        num_classes=1,
        dropout=dropout,
        randflip=randflip,
        block_size=block_size
    )

model_params = [p for p in model.parameters() if p.requires_grad]
# try with weight decay
optim = torch.optim.Adam(params=model_params, lr=lr)
