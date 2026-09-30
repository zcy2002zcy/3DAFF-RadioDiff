"""PEFNet-inspired two-stage field prediction and residual refinement."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from physics import HelmholtzVIE


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 8)
        self.dec3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        mid = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat((F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3), 1))
        d2 = self.dec2(torch.cat((F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2), 1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), 1))
        return self.out(d1)


class AdaptedPEFNet(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        base_channels: int = 32,
        frequency_mhz: float = 3500.0,
        pixel_size_m: float = 1.0,
        contrast_scale: float = 1.0,
        coupling: float = 0.25,
    ) -> None:
        super().__init__()
        self.vie = HelmholtzVIE(
            image_size=image_size,
            frequency_mhz=frequency_mhz,
            pixel_size_m=pixel_size_m,
            contrast_scale=contrast_scale,
            coupling=coupling,
        )
        # q, building occupancy, sparse map, mask, height map, incident real/imaginary
        self.field_net = SmallUNet(7, 2, base_channels)
        self.coarse_head = nn.Sequential(
            ConvBlock(5, base_channels),
            nn.Conv2d(base_channels, 1, 1),
        )
        # coarse map, sparse map, mask, q, building occupancy, height map
        self.refiner = SmallUNet(6, 1, base_channels)

    def freeze_physics(self) -> None:
        for module in (self.field_net, self.coarse_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def freeze_refiner(self) -> None:
        for parameter in self.refiner.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        q = batch["q"]
        building = batch["building"]
        sparse = batch["sparse"]
        mask = batch["mask"]
        height_map = batch["height_map"]

        incident = self.vie.incident_field(q)
        field_input = torch.cat(
            (q, building, sparse, mask, height_map, incident.real, incident.imag), dim=1
        )
        field_delta = self.field_net(field_input)
        field = incident + torch.complex(field_delta[:, :1].float(), field_delta[:, 1:2].float())
        log_magnitude = torch.log(field.abs().clamp_min(1e-6))
        mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
        std = log_magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        normalized_log_magnitude = (log_magnitude - mean) / std

        coarse_logits = self.coarse_head(
            torch.cat((normalized_log_magnitude, q, building, sparse, mask), dim=1)
        )
        coarse = torch.sigmoid(coarse_logits)
        correction = self.refiner(torch.cat((coarse, sparse, mask, q, building, height_map), dim=1))
        refined_raw = torch.sigmoid(torch.logit(coarse.clamp(1e-5, 1.0 - 1e-5)) + correction)
        refined = refined_raw * (1.0 - mask) + sparse * mask
        return {
            "incident": incident,
            "field": field,
            "coarse": coarse,
            "refined_raw": refined_raw,
            "prediction": refined,
        }
