from .. import nn_layers
from torch.nn import functional as F
import torch.nn as nn
import torch
import math

# ===== Neural network building defaults =====
DEFAULT_DTYPE = torch.float32


class Downsample(nn.Module):
    def __init__(self, ch, with_conv):
        super().__init__()
        self.downsample = None
        if with_conv:
            self.downsample = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.downsample(x)
    

class Upsample(nn.Module):
    def __init__(self, ch, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if with_conv:
            self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        if self.with_conv:
            x = self.conv(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1 # Ensure timesteps is 1D

    half_dim = embedding_dim // 2

    # Calculate the sinusoidal base frequency values
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=DEFAULT_DTYPE) * -emb)

    emb = timesteps.to(DEFAULT_DTYPE)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # If embedding_dim is odd, pad with an additional column of zeros
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1)) # Pad the last dimension with one zero column

    assert emb.shape == (timesteps.shape[0], embedding_dim), f"Shape mismatch: {emb.shape}"
    return emb


class Nin(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=1.0):
        super(Nin, self).__init__()

        self.fc = nn.Linear(in_dim, num_units)
        # Apply initialization similar to TensorFlow's custom init
        nn.init.normal_(self.fc.weight, mean=0, std=init_scale)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        original_shape = x.shape #(NCHW)
        x = x.permute(0,2,3,1) #(NHWC)
        # reshape x to [batch_size * height * width, channels]
        x = x.view(-1, original_shape[1])
        x = self.fc(x)
        # reshape back to [batch_size, height, width, num_units]
        x = x.view(original_shape[0], original_shape[2], original_shape[3], -1)
        x = x.permute(0, 3, 1, 2).contiguos() # (NCHW)
        return x


class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch, temb_channels, out_ch=None, conv_shortcut=False, dropout=0.0):
        super(ResnetBlock2D, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch or in_ch
        self.conv_shortcut = conv_shortcut

        # first normalization and convolution
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, stride=1, padding=1)

        # timestep embedding projection
        self.time_emb_proj = nn.Linear(temb_channels, self.out_ch)

        # second normalization, dropout, convolution
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)

        # shortcut connection
        if self.in_ch != self.out_ch:
            if self.conv_shortcut:
                self.shortcut_layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut_layer = Nin(in_dim=in_ch, num_units=out_ch)

    def forward(self, x, temb):
        h = x

        # first path
        h = F.silu(self.norm1(h))
        h = self.conv1(h)

        # add in timestep embedding
        h += self.time_emb_proj(F.silu(temb))[:, :, None, None]

        # second path
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut connection
        if self.in_ch != self.out_ch:
            x = self.shortcut_layer(x)

        return x + h
    

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.q_nin = Nin(in_ch, in_ch)
        self.k_nin = Nin(in_ch, in_ch)
        self.v_nin = Nin(in_ch, in_ch)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.out_proj = Nin(in_ch, in_ch, init_scale=0.0) # verify shape

    def forward(self, x, temb):
        # temb not actually used in attn in this case
        B, C, H, W = x.shape
        
        h = self.norm(x)
        q = self.q_nin(h) # (B, C, H, W)
        k = self.k_nin(h)
        v = self.v_nin(h)

        q = q.permute(0, 2, 3, 1).contiguos().view(B, H * W, C) # (B, H*W, C)
        k = k.permute(0, 2, 3, 1).contiguos().view(B, H * W, C)
        v = v.permute(0, 2, 3, 1).contiguos().view(B, H * W, C)

        # compute attention weights
        w = torch.einsum('bic,bkc->bik', q, k) * (C ** -.5) # (B, HW, HW)
        w = F.softmax(w, dim=-1)

        # apply attention weights
        h = torch.einsum('bik,bkc->bic', w, v) # (B, HW, C)

        # reshape back to the original shape
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)

        # output projection
        h = self.out_proj(h)

        # residual connection
        return x + h
    

class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, out_ch)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_ch, out_ch)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x
    

class DownBlock2D(nn.Module):
    def __init__(self, num_res_blocks, in_ch, out_ch, current_resolution, temb_channels, dropout) -> None:
        super().__init__()
        
        self.resnets = nn.ModuleList()
        for i_block in range(num_res_blocks):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch, temb_channels=temb_channels,
                                              out_ch=out_ch, dropout=dropout))


class UNet2DModel(nn.Module):
    def __init__(self, num_classes, ch, num_res_blocks, initial_resolution, attn_resolutions, out_ch,
                 ch_mult=(1, 2, 4, 8), dropout=0., resamp_with_conv=True, conv_shortcut=False) -> None:
        super().__init__()

        num_resolutions = len(ch_mult)
        assert num_classes == 1, 'not supported'
        self.ch = ch
    
        self.time_embedding = TimestepEmbedding(ch, ch * 4)
        
        # Downsampling
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=self.ch, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()  # Use nn.ModuleList instead of a plain list
        for i_level in range(num_resolutions):
            layers = nn.ModuleList()  # Use nn.ModuleList here as well
            # Calculate the current resolution at this downsampling level
            current_resolution = initial_resolution // 2 ** i_level
            in_ch = ch if (i_level == 0) else ch * ch_mult[i_level - 1]
            # print(f'ch_mult[{i_level}: {ch_mult[i_level]}, current_resolution: {current_resolution}, attn_resolutions: {attn_resolutions}')
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                layers.append(ResnetBlock2D(in_ch=in_ch, temb_channels=ch * 4,
                                 out_ch=ch * ch_mult[i_level], dropout=dropout, conv_shortcut=conv_shortcut))
                if current_resolution in attn_resolutions:
                    layers.append(AttnBlock(ch * ch_mult[i_level]))
            
            if i_level != num_resolutions - 1:
                layers.append(Downsample(ch * ch_mult[i_level], resamp_with_conv))
            self.down_blocks.append(layers)

        # Middle
        self.mid_block1 = ResnetBlock2D(in_ch=ch * ch_mult[i_level], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)
        self.mid_attn = AttnBlock(ch * ch_mult[-1])
        self.mid_block2 = ResnetBlock2D(in_ch=ch * ch_mult[i_level], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()  # Again, use nn.ModuleList
        for i_level in reversed(range(num_resolutions)):
            current_resolution = initial_resolution // 2 ** i_level
            layers = nn.ModuleList()  # Use nn.ModuleList here too
            in_ch = ch * ch_mult[i_level]
            level_out_ch = ch if i_level == 0 else ch * ch_mult[i_level - 1]
            for i_block in range(num_res_blocks + 1):
                layers.append(
                    ResnetBlock2D(in_ch=in_ch, temb_channels= ch * 4, out_ch=level_out_ch,
                                  dropout=dropout, conv_shortcut=conv_shortcut))
                if current_resolution in attn_resolutions:
                    layers.append(AttnBlock(in_ch=level_out_ch))
            if i_level != 0:
                layers.append(Upsample(ch=level_out_ch, with_conv=resamp_with_conv))
            self.up_blocks.append(layers)

        # End
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, y):
        B, C, H, W = x.shape
        assert x.dtype == DEFAULT_DTYPE
        assert t.dtype in [torch.int32, torch.int64]

        temb = get_timestep_embedding(t, self.ch)
        temb = self.time_embedding(temb)
        assert temb.shape == [B, self.ch * 4]

        h = self.conv_in(x)
        hs = [h]

        for block in self.down_blocks:
            for layer in block:
                h = layer(h, temb)
            hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h, temb)
        h = self.mid_block2(h, temb)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h, temb)

        h = self.conv_norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
