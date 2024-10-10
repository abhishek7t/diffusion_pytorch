import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from diffusion_pt.diffusion_utils import GaussianDiffusion, get_beta_schedule
from diffusion_pt.models import unet
from diffusion_pt.gpu_utils.datasets import SimpleDataset, get_dataloader


class Model(nn.Module):
    def __init__(self, *, model_name, betas: np.ndarray, loss_type: str, num_classes: int,
                 dropout: float, block_size: int, device: torch.device):
        super(Model, self).__init__()
        self.model_name = model_name
        self.diffusion = GaussianDiffusion(
            betas=betas, 
            loss_type=loss_type,
            device=device  # Pass the device here
        )
        self.num_classes = num_classes
        self.dropout = dropout
        self.block_size = block_size

        self.unet = unet.UNet2DModel(
            num_classes=num_classes,
            in_channels=3,  # Assuming RGB images
            ch=128,
            num_res_blocks=2,
            initial_resolution=256,
            attn_resolutions=(16,),
            out_ch=3,
            ch_mult=(1, 1, 2, 2, 4, 4),
            dropout=dropout,
            resamp_with_conv=True,
            conv_shortcut=False
        )
        self.unet.to(device)  # Ensure UNet is on the correct device

    def _denoise(self, x, t, y=None, dropout=0.0):
        B, C, H, W = x.shape
        assert x.dtype == torch.float32
        assert t.shape == (B,)

        if self.block_size != 1:
            x = F.pixel_unshuffle(x, self.block_size)

        if self.model_name == 'unet2d16b2c112244':
            out = self.unet(x, t)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")

        if self.block_size != 1:
            out = F.pixel_shuffle(out, self.block_size)

        assert out.shape == (B, C, H, W)
        return out

    def train_fn(self, x: torch.Tensor, y: torch.Tensor):
        B, C, H, W = x.shape

        t = torch.randint(
            low=0, high=self.diffusion.num_timesteps, size=(B,), device=x.device
        )
        losses = self.diffusion.p_losses(
            denoise_fn=lambda x, t: self._denoise(x, t, dropout=self.dropout),
            x_start=x,
            t=t
        )
        assert losses.shape == (B,)
        loss = torch.mean(losses)
        return {'loss': loss}

    def samples_fn(self, shape, device):
        samples = self.diffusion.p_sample_loop(
            denoise_fn=lambda x, t: self._denoise(x, t, dropout=0.0),
            shape=shape,
            device=device
        )
        # Samples are in [-1, 1]; scale to [0, 1] for visualization
        samples = (samples + 1) / 2
        return {'samples': samples}


# Configuration parameters
data_dir = '../datasets/celebahq256/celeba_hq_256/'
torch.manual_seed(5)

dataset_name = 'celebahq256'
optimizer_name = 'adam'
total_bs = 64
grad_clip = 1.0
lr = 0.00002
warmup = 5000
num_diffusion_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
beta_schedule = 'linear'
loss_type = 'noisepred'
dropout = 0.0
block_size = 1
log_dir = 'logs'
num_iterations = 100000
model_name = 'unet2d16b2c112244'

# Memory Optimization Parameters
effective_batch_size = 64  # Desired effective batch size

actual_batch_size = 16
accumulation_steps = effective_batch_size // actual_batch_size # Number of steps to accumulate gradients

model_dir = os.path.join(log_dir, 'models', 'celebhq_diffusion')
os.makedirs(model_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
dataloader = get_dataloader(
    name=dataset_name,
    data_dir=data_dir,
    batch_size=actual_batch_size,
    shuffle=True,
    num_workers=4
)

# Initialize model
model = Model(
    model_name=model_name,
    betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
    ),
    loss_type=loss_type,
    num_classes=1,  # Since CelebA-HQ is unconditional
    dropout=dropout,
    block_size=block_size,
    device=device
)
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Learning rate scheduler with warmup
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) / warmup, 1)
)

# Logging
writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs', model_name))

# Training function
def train(model, dataloader, optimizer, scheduler, num_epochs, device,
          accumulation_steps):
    model.train()
    global_step = 0
    running_loss = 0.0  # To accumulate loss over steps
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            x, y = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss_dict = model.train_fn(x, y)
            loss = loss_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Logging
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step)

            if global_step % 500 == 0:
                print(f"Iteration {global_step}, Loss: {loss.item():.6f}")
                # Save model checkpoint
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{global_step}.pt'))
            if global_step >= num_epochs:
                break
        if global_step >= num_epochs:
            break
    writer.close()

# Evaluation function (optional)
def evaluate(model, num_samples, device):
    model.eval()
    with torch.no_grad():
        samples = model.samples_fn(
            shape=(num_samples, 3, 256, 256), device=device
        )['samples']
        # Clamp and convert samples to [0, 1] for visualization
        samples = torch.clamp(samples, 0, 1)
        # Save or visualize samples as needed
        # Example: save the first sample
        sample_image = transforms.ToPILImage()(samples[0])
        sample_image.save(os.path.join(log_dir, 'sample.png'))
        # Compute Inception Score or FID if implemented
        # Placeholder for metrics computation
        # inception_score = compute_inception_score(samples)
        # print(f"Inception Score: {inception_score}")

# Start training
train(model, dataloader, optimizer, scheduler, num_iterations, device,
      accumulation_steps)

# Optionally evaluate
# evaluate(model, num_samples=64, device=device)
