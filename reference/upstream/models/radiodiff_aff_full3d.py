"""Full 3-D AFF diffusion denoiser for the four-height path-loss volume.

Unlike the legacy height-coupled 2.5-D model, every encoder, bottleneck, and
decoder residual block uses 3 x 3 x 3 convolutions.  Down/up-sampling is
restricted to the horizontal plane so that the four physical receiver-height
layers remain aligned throughout the network.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import DownBlock, ResidualBlock, UpBlock

def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class AdaptiveFusion3D(nn.Module):
    """Learn voxel-wise LoS/depth fusion weights over the complete 3-D volume."""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv3d(2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 2, 1),
        )

    def forward(
        self, los: torch.Tensor, obstruction_depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if los.shape != obstruction_depth.shape or los.ndim != 5:
            raise ValueError("LoS and obstruction depth must share [B,1,D,H,W]")
        sources = torch.cat((los, obstruction_depth), dim=1)
        weights = torch.softmax(self.weight_net(sources), dim=1)
        fused = (weights * sources).sum(dim=1, keepdim=True)
        return fused, weights


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.activation = nn.SiLU()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.block = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        x = self.project(x)
        return self.block(torch.cat((x, skip), dim=1))


class RadioDiffAFFFull3DUNet(nn.Module):
    """Predict DDPM noise for ``[B,1,4,H,W]`` using a full 3-D U-Net."""

    model_type = "pl3d_aff_full3d_ddpm"

    def __init__(self, base_channels: int = 32, aff_hidden_dim: int = 16) -> None:
        super().__init__()
        self.aff = AdaptiveFusion3D(hidden_channels=aff_hidden_dim)
        # xt + {building mask, building height, q, receiver height, distance,
        # AFF-fused LoS/depth, sparse measurement, measurement mask} = 9.
        self.input_block = ResidualBlock3D(9, base_channels)
        self.enc1 = DownBlock3D(base_channels, base_channels * 2)
        self.enc2 = DownBlock3D(base_channels * 2, base_channels * 4)
        bottleneck_channels = base_channels * 4
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(bottleneck_channels, bottleneck_channels),
            ResidualBlock3D(bottleneck_channels, bottleneck_channels),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        )
        self.dec1 = UpBlock3D(bottleneck_channels, base_channels * 2, base_channels * 2)
        self.dec2 = UpBlock3D(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv3d(base_channels, 1, 3, padding=1)
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
            embedding = F.pad(embedding, (0, self.base_channels - embedding.shape[1]))
        return embedding

    def forward(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if xt.ndim != 5 or condition.ndim != 5:
            raise ValueError("xt and condition must be [B,C,D,H,W]")
        if xt.shape[1] != 1 or condition.shape[1] != 9:
            raise ValueError(
                f"Expected xt C=1 and condition C=9, got {xt.shape}, {condition.shape}"
            )
        if xt.shape[2] != 4 or condition.shape[2] != 4:
            raise ValueError("This experiment requires exactly four receiver heights")
        # The AFF branch is also fully 3-D and learns voxel-wise fusion maps.
        geometry, weights = self.aff(
            los=condition[:, 5:6], obstruction_depth=condition[:, 6:7]
        )
        self.last_aff_weight = weights.mean(dim=(-1, -2)).permute(0, 2, 1).detach()

        fixed = torch.cat(
            (condition[:, 0:5], geometry, condition[:, 7:9]),
            dim=1,
        )
        x = torch.cat((xt, fixed), dim=1)
        skip0 = self.input_block(x)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        time_feature = self.time_mlp(self._time_embedding(t))[:, :, None, None, None]
        encoded = self.bottleneck(encoded + time_feature)
        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        return self.output(decoded)


class RadioDiffAFFPrediction3DUNet(RadioDiffAFFFull3DUNet):
    """Strict q+building prediction model with learned 3-D AFF geometry.

    The public dataset condition keeps its stable nine-channel schema, but this
    model can access only building mask, building height, q, LoS, and first-
    obstruction depth.  Sparse path-loss values, sampling masks, IDW priors,
    receiver-height maps, and distance maps are structurally excluded.
    """

    model_type = "pl3d_aff_prediction_q_building_ddpm"

    def __init__(
        self,
        base_channels: int = 32,
        aff_hidden_dim: int = 16,
        aff_mode: str = "adaptive3d",
        use_building_height: bool = True,
    ) -> None:
        super().__init__(base_channels=base_channels, aff_hidden_dim=aff_hidden_dim)
        if aff_mode not in {"adaptive3d", "fixed_mean", "zero"}:
            raise ValueError("aff_mode must be adaptive3d, fixed_mean, or zero")
        self.aff_mode = aff_mode
        self.use_building_height = bool(use_building_height)
        # Full: xt + building mask + building height + q + geometry = 5.
        # No-height ablation: xt + building mask + q + geometry = 4.
        self.input_block = ResidualBlock3D(
            5 if self.use_building_height else 4, base_channels
        )

    def forward(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if xt.ndim != 5 or condition.ndim != 5:
            raise ValueError("xt and condition must be [B,C,D,H,W]")
        if xt.shape[1] != 1 or condition.shape[1] != 9:
            raise ValueError(
                f"Expected xt C=1 and condition C=9, got {xt.shape}, {condition.shape}"
            )
        if xt.shape[2] != 4 or condition.shape[2] != 4:
            raise ValueError("This experiment requires exactly four receiver heights")
        if self.aff_mode == "adaptive3d":
            geometry, weights = self.aff(
                los=condition[:, 5:6], obstruction_depth=condition[:, 6:7]
            )
        elif self.aff_mode == "fixed_mean":
            geometry = 0.5 * (condition[:, 5:6] + condition[:, 6:7])
            weights = torch.full(
                (condition.shape[0], 2, *condition.shape[2:]),
                0.5,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            geometry = torch.zeros_like(condition[:, 5:6])
            weights = torch.zeros(
                (condition.shape[0], 2, *condition.shape[2:]),
                device=condition.device,
                dtype=condition.dtype,
            )
        self.last_aff_weight = weights.mean(dim=(-1, -2)).permute(0, 2, 1).detach()
        building = (
            condition[:, 0:2]
            if self.use_building_height
            else condition[:, 0:1]
        )
        allowed = torch.cat((building, condition[:, 2:3], geometry), dim=1)
        x = torch.cat((xt, allowed), dim=1)
        skip0 = self.input_block(x)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        time_feature = self.time_mlp(self._time_embedding(t))[:, :, None, None, None]
        encoded = self.bottleneck(encoded + time_feature)
        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        return self.output(decoded)


class RadioDiffAFFPrediction2p5DUNet(nn.Module):
    """Strict prediction ablation using shared 2-D blocks and height mixing."""

    model_type = "pl3d_aff_prediction_q_building_2p5d_ddpm"

    def __init__(self, base_channels: int = 32, aff_hidden_dim: int = 16) -> None:
        super().__init__()
        self.aff = AdaptiveFusion3D(hidden_channels=aff_hidden_dim)
        self.input_block = ResidualBlock(
            5, base_channels, norm_type="group", activation="silu"
        )
        self.enc1 = DownBlock(
            base_channels,
            base_channels * 2,
            norm_type="group",
            activation="silu",
        )
        self.enc2 = DownBlock(
            base_channels * 2,
            base_channels * 4,
            norm_type="group",
            activation="silu",
        )
        channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.height_mixer = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.dec1 = UpBlock(
            channels,
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
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.base_channels = base_channels
        self.last_aff_weight: torch.Tensor | None = None

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_channels // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = t.float()[:, None] * frequency[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        return F.pad(embedding, (0, max(0, self.base_channels - embedding.shape[1])))

    def forward(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        batch, _, depth, height, width = xt.shape
        if depth != 4 or condition.shape[1] != 9:
            raise ValueError("Expected xt C=1, condition C=9, and D=4")
        geometry, weights = self.aff(condition[:, 5:6], condition[:, 6:7])
        self.last_aff_weight = weights.mean(dim=(-1, -2)).permute(0, 2, 1).detach()
        fixed = torch.cat((condition[:, 0:3], geometry), dim=1)
        inputs = torch.cat((xt, fixed), dim=1)
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, 5, height, width
        )
        skip0 = self.input_block(inputs)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        _, channels, low_h, low_w = encoded.shape
        time_feature = self.time_mlp(self._time_embedding(t))[:, None].expand(
            batch, depth, channels
        )
        encoded = encoded + time_feature.reshape(batch * depth, channels, 1, 1)
        volume = encoded.reshape(
            batch, depth, channels, low_h, low_w
        ).permute(0, 2, 1, 3, 4)
        volume = F.silu(volume + self.height_mixer(volume))
        encoded = volume.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, low_h, low_w
        )
        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        output = self.output(decoded)
        return output.reshape(batch, depth, 1, height, width).permute(0, 2, 1, 3, 4)


class RadioDiffAFFResidual3DUNet(RadioDiffAFFFull3DUNet):
    """Full-3D AFF denoiser for an IDW-conditioned normalized residual volume."""

    model_type = "pl3d_aff_full3d_idw_residual_v_ddpm"

    def __init__(
        self,
        base_channels: int = 32,
        aff_hidden_dim: int = 16,
        aff_mode: str = "adaptive3d",
    ) -> None:
        super().__init__(base_channels=base_channels, aff_hidden_dim=aff_hidden_dim)
        if aff_mode not in {"adaptive3d", "fixed_mean", "zero"}:
            raise ValueError("aff_mode must be adaptive3d, fixed_mean, or zero")
        self.aff_mode = aff_mode
        # Legacy effective inputs (xt + 8 conditions) plus the IDW coarse prior.
        self.input_block = ResidualBlock3D(10, base_channels)

    def forward(
        self,
        xt: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        idw_prior: torch.Tensor,
    ) -> torch.Tensor:
        if xt.ndim != 5 or condition.ndim != 5 or idw_prior.ndim != 5:
            raise ValueError("xt, condition, and idw_prior must be [B,C,D,H,W]")
        if xt.shape[1] != 1 or condition.shape[1] != 9 or idw_prior.shape[1] != 1:
            raise ValueError("Expected xt C=1, condition C=9, and IDW prior C=1")
        if xt.shape[2] != 4 or condition.shape[2] != 4 or idw_prior.shape[2] != 4:
            raise ValueError("This experiment requires exactly four receiver heights")
        if self.aff_mode == "adaptive3d":
            geometry, weights = self.aff(
                los=condition[:, 5:6], obstruction_depth=condition[:, 6:7]
            )
        elif self.aff_mode == "fixed_mean":
            geometry = 0.5 * (condition[:, 5:6] + condition[:, 6:7])
            weights = torch.full(
                (condition.shape[0], 2, *condition.shape[2:]),
                0.5,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            geometry = torch.zeros_like(condition[:, 5:6])
            weights = torch.zeros(
                (condition.shape[0], 2, *condition.shape[2:]),
                device=condition.device,
                dtype=condition.dtype,
            )
        self.last_aff_weight = weights.mean(dim=(-1, -2)).permute(0, 2, 1).detach()
        fixed = torch.cat(
            (condition[:, 0:5], geometry, condition[:, 7:9], idw_prior), dim=1
        )
        x = torch.cat((xt, fixed), dim=1)
        skip0 = self.input_block(x)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        time_feature = self.time_mlp(self._time_embedding(t))[:, :, None, None, None]
        encoded = self.bottleneck(encoded + time_feature)
        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        return self.output(decoded)


class RadioDiffAFFResidual2p5DUNet(nn.Module):
    """2.5-D ablation: shared 2-D U-Net plus height-only bottleneck mixing."""

    model_type = "pl3d_aff_idw_residual_2p5d_ablation"

    def __init__(self, base_channels: int = 32, aff_hidden_dim: int = 16) -> None:
        super().__init__()
        self.aff = AdaptiveFusion3D(hidden_channels=aff_hidden_dim)
        self.input_block = ResidualBlock(10, base_channels, norm_type="group", activation="silu")
        self.enc1 = DownBlock(base_channels, base_channels * 2, norm_type="group", activation="silu")
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4, norm_type="group", activation="silu")
        channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.height_mixer = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.dec1 = UpBlock(channels, base_channels * 2, base_channels * 2, norm_type="group", activation="silu")
        self.dec2 = UpBlock(base_channels * 2, base_channels, base_channels, norm_type="group", activation="silu")
        self.output = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.base_channels = base_channels
        self.last_aff_weight: torch.Tensor | None = None

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_channels // 2
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = t.float()[:, None] * frequency[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        return F.pad(embedding, (0, max(0, self.base_channels - embedding.shape[1])))

    def forward(
        self,
        xt: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        idw_prior: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, depth, height, width = xt.shape
        if depth != 4 or condition.shape[1] != 9 or idw_prior.shape[1] != 1:
            raise ValueError("Expected xt/idw C=1, condition C=9, and D=4")
        geometry, weights = self.aff(condition[:, 5:6], condition[:, 6:7])
        self.last_aff_weight = weights.mean(dim=(-1, -2)).permute(0, 2, 1).detach()
        fixed = torch.cat((condition[:, 0:5], geometry, condition[:, 7:9], idw_prior), dim=1)
        inputs = torch.cat((xt, fixed), dim=1)
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(batch * depth, 10, height, width)
        skip0 = self.input_block(inputs)
        skip1 = self.enc1(skip0)
        encoded = self.enc2(skip1)
        _, channels, low_h, low_w = encoded.shape
        time_feature = self.time_mlp(self._time_embedding(t))[:, None].expand(batch, depth, channels)
        encoded = encoded + time_feature.reshape(batch * depth, channels, 1, 1)
        volume = encoded.reshape(batch, depth, channels, low_h, low_w).permute(0, 2, 1, 3, 4)
        volume = F.silu(volume + self.height_mixer(volume))
        encoded = volume.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, low_h, low_w)
        decoded = self.dec1(encoded, skip1)
        decoded = self.dec2(decoded, skip0)
        output = self.output(decoded)
        return output.reshape(batch, depth, 1, height, width).permute(0, 2, 1, 3, 4)
