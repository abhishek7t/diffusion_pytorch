import pdb
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
        self.ch = ch
        if with_conv:
            # Added padding=1 to preserve spatial dimensions before stride
            self.downsample = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, ch, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.ch = ch
        if with_conv:
            # Added padding=1 to preserve spatial dimensions after convolution
            self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        if self.with_conv:
            x = self.conv(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    Create sinusoidal timestep embeddings.
    Ensures that all tensors are on the same device as 'timesteps'.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor"

    half_dim = embedding_dim // 2

    # Calculate the sinusoidal base frequency values
    emb_scale = math.log(10000) / (half_dim - 1)

    # Create 'emb' on the same device and dtype as 'timesteps'
    emb = torch.exp(
        torch.arange(half_dim, dtype=timesteps.dtype, device=timesteps.device) * -emb_scale
    )

    emb = timesteps.to(timesteps.dtype)[:, None] * emb[None, :]  # (N, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)       # (N, embedding_dim)

    # If embedding_dim is odd, pad with an additional column of zeros
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))  # Pad the last dimension with one zero column

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
        N, C, H, W = x.shape  # (NCHW)
        x = x.permute(0, 2, 3, 1).contiguous()  # (NHWC)
        # reshape x to [batch_size * height * width, channels]
        x = x.view(-1, C)
        x = self.fc(x)
        # reshape back to [batch_size, height, width, num_units]
        x = x.view(N, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # (NCHW)
        return x


class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch, temb_channels, out_ch=None, conv_shortcut=False, dropout=0.0):
        super(ResnetBlock2D, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch or in_ch
        self.conv_shortcut = conv_shortcut

        # First normalization and convolution
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, stride=1, padding=1)

        # Timestep embedding projection
        self.time_emb_proj = nn.Linear(temb_channels, self.out_ch)

        # Second normalization, dropout, convolution
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=1, padding=1)

        # Shortcut connection
        if self.in_ch != self.out_ch:
            if self.conv_shortcut:
                self.shortcut_layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut_layer = Nin(in_dim=in_ch, num_units=self.out_ch)

    def forward(self, x, temb):
        h = x
        assert x.shape[1] == self.in_ch, f"x channels: {x.shape[1]}, self.in_ch: {self.in_ch} should be equal before norm1"

        # First path
        h = F.silu(self.norm1(h))
        h = self.conv1(h)

        # Add in timestep embedding
        h += self.time_emb_proj(F.silu(temb))[:, :, None, None]

        # Second path
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # Shortcut connection
        if self.in_ch != self.out_ch:
            x = self.shortcut_layer(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.to_q = Nin(in_ch, in_ch)
        self.to_k = Nin(in_ch, in_ch)
        self.to_v = Nin(in_ch, in_ch)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.to_out = Nin(in_ch, in_ch, init_scale=0.0)  # Verify shape

    def forward(self, x, temb):
        # temb not actually used in attention in this case
        B, C, H, W = x.shape

        h = self.norm(x)
        q = self.to_q(h)  # (B, C, H, W)
        k = self.to_k(h)
        v = self.to_v(h)

        q = q.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, H*W, C)
        k = k.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        v = v.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # Compute attention weights
        w = torch.einsum('bic,bkc->bik', q, k) * (C ** -0.5)  # (B, HW, HW)
        w = F.softmax(w, dim=-1)

        # Apply attention weights
        h = torch.einsum('bik,bkc->bic', w, v)  # (B, HW, C)

        # Reshape back to the original shape
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Output projection
        h = self.to_out(h)

        # Residual connection
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
    def __init__(self, in_ch, out_ch, temb_channels, num_res_blocks, num_resolutions, 
                 i_level, dropout, conv_shortcut, resamp_with_conv):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i_block in range(num_res_blocks):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch, temb_channels=temb_channels,
                                out_ch=out_ch, dropout=dropout, conv_shortcut=conv_shortcut))
            in_ch = out_ch
        
        self.downsamplers = nn.ModuleList()
        if i_level != num_resolutions - 1: # no downsampling at the end of last down block
            self.downsamplers.append(Downsample(out_ch, resamp_with_conv))
        
    def forward(self, x, temb):
        hs = []
        for res_block in self.resnets:
            x = res_block(x, temb)
            hs.append(x)
            
        for downsampler in self.downsamplers:
            x = downsampler(x)
            hs.append(x)
        
        return x, hs


class AttnDownBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_channels, num_res_blocks, num_resolutions, 
                 i_level, dropout, conv_shortcut, resamp_with_conv):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i_block in range(num_res_blocks):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch, temb_channels=temb_channels,
                                out_ch=out_ch, dropout=dropout, conv_shortcut=conv_shortcut))
            in_ch = out_ch
            
        for i_block in range(num_res_blocks):
            self.attentions.append(AttnBlock(out_ch))
        
        if i_level != num_resolutions - 1: # no downsampling at the end of last down block
            self.downsamplers.append(Downsample(out_ch, resamp_with_conv))
        
    def forward(self, x, temb):
        hs = []
        for res_block, attn_block in zip(self.resnets, self.attentions):
            x = res_block(x, temb)
            x = attn_block(x, temb)
            hs.append(x)
            
        for downsampler in self.downsamplers:
            x = downsampler(x)
            hs.append(x)
        return x, hs


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, skip_conn_ch, out_ch, temb_channels, num_res_blocks,
                 num_resolutions, i_level, dropout, conv_shortcut, resamp_with_conv):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        assert num_res_blocks + 1 == len(skip_conn_ch), f"i_level: {i_level}, num_res_blocks: {num_res_blocks}, skip_conn_ch: {skip_conn_ch}"
        for i in range(num_res_blocks + 1):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch + skip_conn_ch[i], temb_channels= temb_channels,
                            out_ch=out_ch, dropout=dropout, conv_shortcut=conv_shortcut))
            in_ch = out_ch
            
        if i_level != 0: # no upsampling at the end of last up block
            self.upsamplers.append(Upsample(ch=out_ch, with_conv=resamp_with_conv))
        
    def forward(self, x, skip_conn_hs, temb):
        for i_block, res_block in enumerate(self.resnets):
            x = res_block(torch.cat([x, skip_conn_hs[i_block]], dim=1), temb)
            
        for upsampler in self.upsamplers:
            x = upsampler(x)
        
        return x


class AttnUpBlock2D(nn.Module):
    def __init__(self, in_ch, skip_conn_ch, out_ch, temb_channels, num_res_blocks,
                 num_resolutions, i_level, dropout, conv_shortcut, resamp_with_conv):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        assert num_res_blocks + 1 == len(skip_conn_ch), f"i_level: {i_level}, num_res_blocks: {num_res_blocks}, skip_conn_ch: {skip_conn_ch}"
        for i in range(num_res_blocks + 1):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch + skip_conn_ch[i], temb_channels= temb_channels,
                            out_ch=out_ch, dropout=dropout, conv_shortcut=conv_shortcut))
            self.attentions.append(AttnBlock(out_ch))
            in_ch = out_ch
            
        if i_level != 0: # no upsampling at the end of last up block
            self.upsamplers.append(Upsample(ch=out_ch, with_conv=resamp_with_conv))
        
    def forward(self, x, skip_conn_hs, temb):
        for i_block, (res_block, attn_block) in enumerate(zip(self.resnets, self.attentions)):
            x = res_block(torch.cat([x, skip_conn_hs[i_block]], dim=1), temb)
            x = attn_block(x, temb)
            
        for upsampler in self.upsamplers:
            x = upsampler(x)
        
        return x


class UNet2DModel(nn.Module):
    def __init__(self, num_classes, in_channels, ch, num_res_blocks, initial_resolution, attn_resolutions, out_ch,
                 ch_mult=(1, 2, 4, 8), dropout=0., resamp_with_conv=True, conv_shortcut=False) -> None:
        super().__init__()

        self.num_resolutions = len(ch_mult)
        assert num_classes == 1, 'Only unconditional models are supported.'
        self.ch = ch
        self.num_res_blocks = num_res_blocks

        self.time_embedding = TimestepEmbedding(ch, ch * 4)

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()  # Use nn.ModuleList instead of a plain list
        in_ch = self.conv_in.out_channels
        for i_level in range(self.num_resolutions):
            # Calculate the current resolution at this downsampling level
            current_resolution = initial_resolution // 2 ** i_level
            if current_resolution in attn_resolutions:
                self.down_blocks.append(AttnDownBlock2D(in_ch=in_ch, out_ch=ch * ch_mult[i_level],
                                                        temb_channels=ch * 4, num_res_blocks=num_res_blocks,
                                                        num_resolutions=self.num_resolutions, i_level=i_level,
                                                        dropout=dropout, conv_shortcut=conv_shortcut,
                                                        resamp_with_conv=resamp_with_conv))
            else:
                self.down_blocks.append(DownBlock2D(in_ch=in_ch, out_ch=ch * ch_mult[i_level],
                                                        temb_channels=ch * 4, num_res_blocks=num_res_blocks,
                                                        num_resolutions=self.num_resolutions, i_level=i_level,
                                                        dropout=dropout, conv_shortcut=conv_shortcut,
                                                        resamp_with_conv=resamp_with_conv))
            in_ch = ch * ch_mult[i_level]

        # Middle
        self.mid_block1 = ResnetBlock2D(in_ch=ch * ch_mult[-1], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)
        self.mid_attn = AttnBlock(ch * ch_mult[-1])
        self.mid_block2 = ResnetBlock2D(in_ch=ch * ch_mult[-1], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        in_ch = self.mid_block2.out_ch
        for i_level in reversed(range(self.num_resolutions)):
            current_resolution = initial_resolution // 2 ** i_level
            level_out_ch = ch if i_level == 0 else ch * ch_mult[i_level]            
            skip_conn_ch = [self.down_blocks[i_level].resnets[j].out_ch for j in reversed(range(num_res_blocks))]
            if i_level != 0:
                skip_conn_ch.append(self.down_blocks[i_level - 1].downsamplers[-1].ch)
            else:
                skip_conn_ch.append(self.ch)

            if current_resolution in attn_resolutions:
                self.up_blocks.append(AttnUpBlock2D(in_ch=in_ch, skip_conn_ch=skip_conn_ch,
                                                    out_ch=level_out_ch, num_res_blocks=num_res_blocks,
                                                    num_resolutions=self.num_resolutions, i_level=i_level,
                                                    dropout=dropout, conv_shortcut=conv_shortcut,
                                                    resamp_with_conv=resamp_with_conv, temb_channels=ch * 4))
            else:
                self.up_blocks.append(UpBlock2D(in_ch=in_ch, skip_conn_ch=skip_conn_ch,
                                                out_ch=level_out_ch, num_res_blocks=num_res_blocks,
                                                num_resolutions=self.num_resolutions, i_level=i_level,
                                                dropout=dropout, conv_shortcut=conv_shortcut,
                                                resamp_with_conv=resamp_with_conv, temb_channels=ch * 4))
            in_ch = ch * ch_mult[i_level]

        # End
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, y=None):
        B, C, H, W = x.shape
        assert x.dtype == DEFAULT_DTYPE, f"Expected x dtype {DEFAULT_DTYPE}, but got {x.dtype}"
        assert t.dtype in [torch.int32, torch.int64], f"Expected t dtype int32 or int64, but got {t.dtype}"

        temb = get_timestep_embedding(t, self.ch)
        temb = self.time_embedding(temb)

        # Corrected assertion: use tuple instead of list
        expected_shape = (B, self.ch * 4)
        assert temb.shape == expected_shape, f"Expected temb shape {expected_shape}, but got {temb.shape}"

        h = self.conv_in(x)
        h_conv_in = h
        
        hs = []
        for i_level, block in enumerate(self.down_blocks):
            h, hs_i_level = block(h, temb)
            hs.append(hs_i_level)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h, temb)
        h = self.mid_block2(h, temb)
   
        for i, block in enumerate(self.up_blocks):
            i_level = self.num_resolutions - i - 1
            hs_skip_conn = hs[i_level][:self.num_res_blocks]
            if i_level != 0:
                hs_skip_conn.append(hs[i_level - 1][-1])
            else:
                hs_skip_conn.append(h_conv_in)
            h = block(h, hs_skip_conn, temb)

        h = self.conv_norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
