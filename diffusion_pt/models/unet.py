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

        self.q_nin = Nin(in_ch, in_ch)
        self.k_nin = Nin(in_ch, in_ch)
        self.v_nin = Nin(in_ch, in_ch)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.out_proj = Nin(in_ch, in_ch, init_scale=0.0)  # Verify shape

    def forward(self, x, temb):
        # temb not actually used in attention in this case
        B, C, H, W = x.shape

        h = self.norm(x)
        q = self.q_nin(h)  # (B, C, H, W)
        k = self.k_nin(h)
        v = self.v_nin(h)

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
        h = self.out_proj(h)

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
    def __init__(self, num_res_blocks, in_ch, out_ch, current_resolution, temb_channels, dropout) -> None:
        super().__init__()

        self.resnets = nn.ModuleList()
        for i_block in range(num_res_blocks):
            self.resnets.append(ResnetBlock2D(in_ch=in_ch, temb_channels=temb_channels,
                                              out_ch=out_ch, dropout=dropout))


class UNet2DModel(nn.Module):
    def __init__(self, num_classes, in_channels, ch, num_res_blocks, initial_resolution, attn_resolutions, out_ch,
                 ch_mult=(1, 2, 4, 8), dropout=0., resamp_with_conv=True, conv_shortcut=False) -> None:
        super().__init__()

        num_resolutions = len(ch_mult)
        assert num_classes == 1, 'Only unconditional models are supported.'
        self.ch = ch

        self.time_embedding = TimestepEmbedding(ch, ch * 4)

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()  # Use nn.ModuleList instead of a plain list
        in_ch = self.conv_in.out_channels
        for i_level in range(num_resolutions):
            layers = nn.ModuleList()  # Use nn.ModuleList here as well
            # Calculate the current resolution at this downsampling level
            current_resolution = initial_resolution // 2 ** i_level
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                layers.append(ResnetBlock2D(in_ch=in_ch, temb_channels=ch * 4,
                                 out_ch=ch * ch_mult[i_level], dropout=dropout, conv_shortcut=conv_shortcut))
                in_ch = layers[-1].out_ch
                if current_resolution in attn_resolutions:
                    layers.append(AttnBlock(ch * ch_mult[i_level]))

            if i_level != num_resolutions - 1:
                layers.append(Downsample(ch * ch_mult[i_level], resamp_with_conv))
            self.down_blocks.append(layers)

        # Middle
        self.mid_block1 = ResnetBlock2D(in_ch=ch * ch_mult[-1], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)
        self.mid_attn = AttnBlock(ch * ch_mult[-1])
        self.mid_block2 = ResnetBlock2D(in_ch=ch * ch_mult[-1], temb_channels=ch * 4,
            out_ch=ch * ch_mult[-1], dropout=dropout, conv_shortcut=conv_shortcut)

        # Upsampling
        self.up_blocks = nn.ModuleList()  # Again, use nn.ModuleList
        in_ch = 2 * self.mid_block2.out_ch
        for i_level in reversed(range(num_resolutions)):
            current_resolution = initial_resolution // 2 ** i_level
            layers = nn.ModuleList()  # Use nn.ModuleList here too
            # in_ch = ch * ch_mult[i_level]
            level_out_ch = ch if i_level == 0 else ch * ch_mult[i_level - 1]
            for i_block in range(num_res_blocks + 1):
                layers.append(
                    ResnetBlock2D(in_ch=in_ch, temb_channels= ch * 4, out_ch=level_out_ch,
                                  dropout=dropout, conv_shortcut=conv_shortcut))
                in_ch = layers[-1].out_ch
                if current_resolution in attn_resolutions:
                    layers.append(AttnBlock(in_ch=level_out_ch))
            in_ch = 2 * level_out_ch
            if i_level != 0:
                layers.append(Upsample(ch=level_out_ch, with_conv=resamp_with_conv))
            self.up_blocks.append(layers)

        # End
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, y=None):
        B, C, H, W = x.shape
        assert x.dtype == DEFAULT_DTYPE, f"Expected x dtype {DEFAULT_DTYPE}, but got {x.dtype}"
        assert t.dtype in [torch.int32, torch.int64], f"Expected t dtype int32 or int64, but got {t.dtype}"

        # Debug: Check device of 'x' and 't'
        print(f"x device: {x.device}, t device: {t.device}")

        temb = get_timestep_embedding(t, self.ch)
        print(f"emb device after get_timestep_embedding: {temb.device}")  # Should match 't.device'

        temb = self.time_embedding(temb)
        print(f"temb device after time_embedding: {temb.device}")  # Should match 'x.device'

        # Corrected assertion: use tuple instead of list
        expected_shape = (B, self.ch * 4)
        assert temb.shape == expected_shape, f"Expected temb shape {expected_shape}, but got {temb.shape}"

        print(f"x shape before: {x.shape}")
        h = self.conv_in(x)
        print(f"h shape after conv_in: {h.shape}")
        hs = [h]

        for block_ind, block in enumerate(self.down_blocks):
            for layer_ind, layer in enumerate(block):
                # print(f"block: {block_ind}, layer: {layer_ind}, layer: {type(layer)}, h shape before: {h.shape}")
                if isinstance(layer, Downsample):
                    h = layer(h)
                else:
                    h = layer(h, temb)
                # print(f"block: {block_ind}, layer: {layer_ind}, layer: {type(layer)}, h shape after: {h.shape}")
            hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h, temb)
        h = self.mid_block2(h, temb)
        print(f"mid_block2 in: {self.mid_block2.in_ch}, h shape after: {h.shape}")

        for block_ind, block in enumerate(self.up_blocks):
            hs_connection = hs.pop()
            print(f"block_ind: {block_ind}, hs_connection: {hs_connection.shape}, h shape before: {h.shape}")
            h = torch.cat([h, hs_connection], dim=1)
            for layer_ind, layer in enumerate(block):
                print(f"block: {block_ind}, layer: {layer_ind}, layer: {type(layer)}, h shape before: {h.shape}")
                if isinstance(layer, Upsample):
                    h = layer(h)
                else:
                    h = layer(h, temb)
                print(f"block: {block_ind}, layer: {layer_ind}, layer: {type(layer)}, h shape after: {h.shape}")

        h = self.conv_norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
