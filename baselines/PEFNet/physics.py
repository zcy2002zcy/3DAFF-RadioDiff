"""Differentiable 2-D Helmholtz volume-integral-equation operators."""

from __future__ import annotations

import math

import torch
from torch import nn


class HelmholtzVIE(nn.Module):
    """Discretized scalar VIE: E = E_inc + alpha * G * (chi E).

    The source paper uses environmental material information for ``chi``. The
    local dataset has no complex permittivity, so receiver-height building
    occupancy rasterized from raw polygons is used as a documented proxy.
    """

    def __init__(
        self,
        image_size: int = 256,
        frequency_mhz: float = 3500.0,
        pixel_size_m: float = 1.0,
        contrast_scale: float = 1.0,
        coupling: float = 0.25,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.frequency_mhz = float(frequency_mhz)
        self.pixel_size_m = float(pixel_size_m)
        self.contrast_scale = float(contrast_scale)
        self.coupling = float(coupling)

        speed = 299_792_458.0
        wavelength = speed / (self.frequency_mhz * 1e6)
        self.wave_number = 2.0 * math.pi / wavelength

        coords = torch.arange(self.image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("yy", yy, persistent=False)
        self.register_buffer("xx", xx, persistent=False)

        periodic = torch.minimum(coords, self.image_size - coords) * self.pixel_size_m
        ky, kx = torch.meshgrid(periodic, periodic, indexing="ij")
        radius = torch.sqrt(kx.square() + ky.square()).clamp_min(self.pixel_size_m * 0.5)
        amplitude = torch.rsqrt(radius)
        kernel = amplitude * torch.exp(1j * self.wave_number * radius)
        kernel[0, 0] = 0.0
        kernel = kernel / kernel.abs().sum().clamp_min(1e-8)
        self.register_buffer("green_fft", torch.fft.fft2(kernel.to(torch.complex64)), persistent=False)

    def incident_field(self, q: torch.Tensor) -> torch.Tensor:
        q = q.float()
        batch, _, height, width = q.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected {self.image_size}x{self.image_size}, got {height}x{width}")
        source = q.flatten(2).argmax(dim=-1)
        source_y = (source // width).view(batch, 1, 1)
        source_x = (source % width).view(batch, 1, 1)
        radius = torch.sqrt(
            (self.yy[None] - source_y).square() + (self.xx[None] - source_x).square()
        ) * self.pixel_size_m
        radius = radius.clamp_min(self.pixel_size_m * 0.5)
        amplitude = torch.rsqrt(radius)
        amplitude = amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        field = amplitude * torch.exp(1j * self.wave_number * radius)
        return field[:, None].to(torch.complex64)

    def susceptibility(self, building: torch.Tensor) -> torch.Tensor:
        return building.float().clamp(0.0, 1.0) * self.contrast_scale

    def scatter(self, field: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
        contrast = self.susceptibility(building).to(torch.complex64)
        polarization = contrast * field.to(torch.complex64)
        return torch.fft.ifft2(torch.fft.fft2(polarization) * self.green_fft[None, None])

    def residual(self, field: torch.Tensor, incident: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
        rhs = incident + self.coupling * self.scatter(field, building)
        return field.to(torch.complex64) - rhs

    def loss(self, field: torch.Tensor, incident: torch.Tensor, building: torch.Tensor) -> torch.Tensor:
        residual = self.residual(field, incident, building)
        numerator = residual.abs().square().mean(dim=(-3, -2, -1))
        denominator = incident.abs().square().mean(dim=(-3, -2, -1)).clamp_min(1e-8)
        return (numerator / denominator).mean()
