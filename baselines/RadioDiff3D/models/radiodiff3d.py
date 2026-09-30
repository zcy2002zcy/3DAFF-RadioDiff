import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=t.device, dtype=torch.float32)
        )
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        return F.pad(embedding, (0, self.dim - embedding.shape[1]))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(time_embedding).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None])
        h = h + shift[:, :, None, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class BottleneckAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 4, 4),
            stride=(1, 2, 2), padding=(0, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.conv(x)


class RadioDiff3D(nn.Module):
    """Conditional 3D residual U-Net noise predictor adapted from the paper description."""

    def __init__(
        self,
        in_channels: int = 1,
        condition_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 16,
        time_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        b = base_channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_conv = nn.Conv3d(in_channels + condition_channels, b, 3, padding=1)

        self.enc0 = ResBlock3D(b, b, time_dim)
        self.down0 = Downsample3D(b, b * 2)
        self.enc1 = ResBlock3D(b * 2, b * 2, time_dim)
        self.down1 = Downsample3D(b * 2, b * 4)
        self.enc2 = ResBlock3D(b * 4, b * 4, time_dim)
        self.down2 = Downsample3D(b * 4, b * 8)
        self.enc3 = ResBlock3D(b * 8, b * 8, time_dim)
        self.down3 = Downsample3D(b * 8, b * 8)

        self.mid1 = ResBlock3D(b * 8, b * 8, time_dim)
        self.mid_attention = BottleneckAttention3D(b * 8, attention_heads)
        self.mid2 = ResBlock3D(b * 8, b * 8, time_dim)

        self.up3 = Upsample3D(b * 8, b * 8)
        self.dec3 = ResBlock3D(b * 16, b * 8, time_dim)
        self.up2 = Upsample3D(b * 8, b * 4)
        self.dec2 = ResBlock3D(b * 8, b * 4, time_dim)
        self.up1 = Upsample3D(b * 4, b * 2)
        self.dec1 = ResBlock3D(b * 4, b * 2, time_dim)
        self.up0 = Upsample3D(b * 2, b)
        self.dec0 = ResBlock3D(b * 2, b, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(b), b), nn.SiLU(), nn.Conv3d(b, out_channels, 3, padding=1)
        )

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(t)
        x0 = self.enc0(self.input_conv(torch.cat([x_t, condition], dim=1)), time_embedding)
        x1 = self.enc1(self.down0(x0), time_embedding)
        x2 = self.enc2(self.down1(x1), time_embedding)
        x3 = self.enc3(self.down2(x2), time_embedding)
        x = self.mid1(self.down3(x3), time_embedding)
        x = self.mid_attention(x)
        x = self.mid2(x, time_embedding)
        x = self.dec3(torch.cat([self.up3(x), x3], dim=1), time_embedding)
        x = self.dec2(torch.cat([self.up2(x), x2], dim=1), time_embedding)
        x = self.dec1(torch.cat([self.up1(x), x1], dim=1), time_embedding)
        x = self.dec0(torch.cat([self.up0(x), x0], dim=1), time_embedding)
        return self.output(x)

