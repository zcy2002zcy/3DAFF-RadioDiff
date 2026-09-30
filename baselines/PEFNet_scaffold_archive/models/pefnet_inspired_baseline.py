"""Independent PEFNet-inspired two-stage U-Net adaptation.

This module follows the paper's knowledge-driven complex-field prediction and
data-driven residual refinement, but it is not upstream PEFNet code.  The
caller must combine ``total_field`` with :class:`physics.VIEOperator` and must
label results as an adapted/inspired baseline.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int, max_groups: int) -> int:
    """Choose a deterministic GroupNorm partition with >=4 channels/group."""

    if channels <= 0 or max_groups <= 0:
        raise ValueError("channels and max_groups must be positive")
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0 and channels // groups >= 4:
            return groups
    return 1


class _ConvBlock(nn.Module):
    """Two 3x3 Conv-GroupNorm-Tanh operations.

    The paper diagram uses normalization but does not publish training code.
    GroupNorm is the documented adaptation used here because the formal dense
    baseline has batch size one; BatchNorm running statistics were empirically
    unstable under that protocol.
    """

    def __init__(
        self, in_channels: int, out_channels: int, *, max_norm_groups: int
    ) -> None:
        super().__init__()
        groups = _group_count(out_channels, max_norm_groups)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class _PaperUNet(nn.Module):
    """Bilinear-upsample U-Net matching the architecture family in PEFNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int,
        levels: int,
        final_tanh: bool = True,
        max_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) <= 0:
            raise ValueError("channel counts must be positive")
        if levels < 2:
            raise ValueError("levels must be at least 2")
        widths = [base_channels * (2**index) for index in range(levels)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for width in widths:
            self.encoders.append(
                _ConvBlock(current, width, max_norm_groups=max_norm_groups)
            )
            current = width
        self.pool = nn.MaxPool2d(2)
        self.up_projections = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up_projections.append(nn.Conv2d(current, width, 3, padding=1))
            self.decoders.append(
                _ConvBlock(width * 2, width, max_norm_groups=max_norm_groups)
            )
            current = width
        # Figure 3 narrows the top-level decoder feature map once more before
        # producing the final channels (128 -> 64 -> 2 in the paper setting).
        head_channels = max(base_channels // 2, 1)
        self.output = nn.Sequential(
            nn.Conv2d(current, head_channels, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(head_channels, out_channels, 3, padding=1),
        )
        self.output_activation: nn.Module = nn.Tanh() if final_tanh else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        skips: list[Tensor] = []
        value = inputs
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            if index < len(self.encoders) - 1:
                skips.append(value)
                value = self.pool(value)
        for projection, decoder, skip in zip(
            self.up_projections, self.decoders, reversed(skips)
        ):
            value = F.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            value = projection(value)
            value = decoder(torch.cat((value, skip), dim=1))
        return self.output_activation(self.output(value))


class PEFNetInspiredBaseline(nn.Module):
    """Predict a complex total field and refine its dB-like attenuation map."""

    def __init__(
        self,
        in_channels: int = 5,
        *,
        base_channels: int = 128,
        levels: int = 4,
        amplitude_epsilon: float = 1e-6,
        initial_rough_scale: float = 0.01,
        max_norm_groups: int = 32,
        force_refinement_fp32: bool = True,
    ) -> None:
        super().__init__()
        if amplitude_epsilon <= 0:
            raise ValueError("amplitude_epsilon must be positive")
        if initial_rough_scale <= 0:
            raise ValueError("initial_rough_scale must be positive")
        if max_norm_groups <= 0:
            raise ValueError("max_norm_groups must be positive")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.levels = int(levels)
        self.amplitude_epsilon = float(amplitude_epsilon)
        self.max_norm_groups = int(max_norm_groups)
        self.force_refinement_fp32 = bool(force_refinement_fp32)
        self.knowledge_unet = _PaperUNet(
            self.in_channels,
            2,
            base_channels=self.base_channels,
            levels=self.levels,
            final_tanh=True,
            max_norm_groups=self.max_norm_groups,
        )
        self.refinement_unet = _PaperUNet(
            1,
            1,
            base_channels=self.base_channels,
            levels=self.levels,
            final_tanh=True,
            max_norm_groups=self.max_norm_groups,
        )
        # The paper does not specify the exact electric-field-to-normalized-PL
        # calibration.  A positive learnable affine conversion is explicit,
        # differentiable, and trained only from the training split.
        self.log_rough_scale = nn.Parameter(
            torch.tensor(math.log(initial_rough_scale), dtype=torch.float32)
        )
        self.rough_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def field_to_rough_pathloss(self, total_field: Tensor) -> Tensor:
        if total_field.ndim != 4 or total_field.shape[1] != 2:
            raise ValueError(
                f"total_field must have shape [B,2,H,W], got {tuple(total_field.shape)}"
            )
        # Keep the logarithmic field-to-attenuation conversion in FP32.  With
        # AMP, epsilon**2 underflows in FP16 and can create infinite gradients
        # around a zero predicted field even when the scalar loss is finite.
        field = total_field.float()
        real, imaginary = field[:, 0:1], field[:, 1:2]
        amplitude = torch.sqrt(
            real.square() + imaginary.square() + self.amplitude_epsilon**2
        )
        attenuation_db_like = -20.0 * torch.log10(amplitude + self.amplitude_epsilon)
        scale = self.log_rough_scale.exp()
        return scale * attenuation_db_like + self.rough_bias

    def forward_components(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"inputs must have shape [B,{self.in_channels},H,W], got {tuple(inputs.shape)}"
            )
        total_field = self.knowledge_unet(inputs)
        # BF16 is safe for the first, bounded field U-Net, but the logarithmic
        # conversion and refinement branch are deliberately FP32.  This avoids
        # the FP16/BF16 autocast context leaking into the branch that previously
        # developed exploding normalization statistics.
        with torch.autocast(
            device_type=inputs.device.type,
            enabled=False if self.force_refinement_fp32 else torch.is_autocast_enabled(),
        ):
            rough_pathloss = self.field_to_rough_pathloss(total_field.float())
            residual = self.refinement_unet(rough_pathloss)
            prediction = rough_pathloss + residual
        return {
            "prediction": prediction,
            "total_field": total_field,
            "rough_pathloss": rough_pathloss,
            "residual": residual,
        }

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        return self.forward_components(inputs)


def build_pefnet_inspired(
    in_channels: int = 5,
    base_channels: int = 128,
    *,
    levels: int = 4,
    max_norm_groups: int = 32,
    force_refinement_fp32: bool = True,
) -> PEFNetInspiredBaseline:
    """Build the independently implemented adapted baseline."""

    return PEFNetInspiredBaseline(
        in_channels=in_channels,
        base_channels=base_channels,
        levels=levels,
        max_norm_groups=max_norm_groups,
        force_refinement_fp32=force_refinement_fp32,
    )


__all__ = ["PEFNetInspiredBaseline", "build_pefnet_inspired"]
