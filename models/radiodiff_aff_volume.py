"""Memory-conscious h1--h4 AFF diffusion denoiser.

Spatial encoding is shared across receiver heights.  A learnable bottleneck
mixes features only along the height axis, giving cross-height interaction
without the memory cost of a full-resolution 3-D U-Net.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.adaptive_fusion import AdaptiveFusion
from models.blocks import DownBlock, ResidualBlock, UpBlock


class RadioDiffAFFVolumeUNet(nn.Module):
    """Predict diffusion noise for tensors shaped ``[B,1,4,H,W]``."""

    def __init__(self, base_channels: int = 32, aff_hidden_dim: int = 16) -> None:
        super().__init__()
        self.aff = AdaptiveFusion(hidden_dim=aff_hidden_dim)
        # xt + {building mask, building height, q, rx height, distance,
        # fused geometry, sparse measurement, measurement mask}
        self.input_block = ResidualBlock(9, base_channels, norm_type="group", activation="silu")
        self.enc1 = DownBlock(base_channels, base_channels * 2, norm_type="group", activation="silu")
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4, norm_type="group", activation="silu")
        bottleneck_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        )
        self.height_mixer = nn.Sequential(
            nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.GroupNorm(8, bottleneck_channels),
            nn.SiLU(),
            nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.GroupNorm(8, bottleneck_channels),
        )
        self.mix_activation = nn.SiLU()
        self.dec1 = UpBlock(
            bottleneck_channels,
            base_channels * 2,
            base_channels * 2,
            norm_type="group",
            activation="silu",
        )
        self.dec2 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            norm_type="group",
            activation="silu",
        )
        self.output = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.last_aff_weight: torch.Tensor | None = None
        self.base_channels = base_channels

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_channels // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = t.float()[:, None] * frequency[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        if embedding.shape[1] < self.base_channels:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.base_channels - embedding.shape[1])
            )
        return embedding

    def forward(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if xt.ndim != 5 or condition.ndim != 5:
            raise ValueError("xt and condition must be [B,C,D,H,W]")
        if xt.shape[1] != 1 or condition.shape[1] != 9:
            raise ValueError(f"Expected xt C=1 and condition C=9, got {xt.shape}, {condition.shape}")
        if xt.shape[2] != 4 or condition.shape[2] != 4:
            raise ValueError("This experiment requires exactly four receiver heights")
        batch, _, depth, height, width = xt.shape

        los = condition[:, 5].reshape(batch * depth, 1, height, width)
        obstruction_depth = condition[:, 6].reshape(batch * depth, 1, height, width)
        geometry, weights = self.aff(los=los, depth=obstruction_depth)
        self.last_aff_weight = weights.reshape(batch, depth, 2).detach()

        fixed = torch.cat(
            (
                condition[:, 0:5],
                geometry.reshape(batch, depth, height, width)[:, None],
                condition[:, 7:9],
            ),
            dim=1,
        )
        inputs = torch.cat((xt, fixed), dim=1)
        inputs_2d = inputs.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, 9, height, width
        )

        skip0 = self.input_block(inputs_2d)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        _, channels, low_h, low_w = encoded.shape
        time_feature = self.time_mlp(self._time_embedding(t))
        time_feature = time_feature[:, None].expand(batch, depth, channels)
        encoded = encoded + time_feature.reshape(batch * depth, channels, 1, 1)
        volume = encoded.reshape(batch, depth, channels, low_h, low_w).permute(0, 2, 1, 3, 4)
        volume = self.mix_activation(volume + self.height_mixer(volume))
        encoded = volume.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, low_h, low_w
        )

        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        output = self.output(decoded)
        return output.reshape(batch, depth, 1, height, width).permute(0, 2, 1, 3, 4)

