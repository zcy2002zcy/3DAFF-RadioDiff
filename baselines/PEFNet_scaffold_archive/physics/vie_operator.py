"""Differentiable FFT implementation of the adapted two-dimensional VIE loss.

The source PEFNet paper writes the discretized volume integral equation as

    (I + W chi) E_tot = E_inc.

Forming ``W`` densely is infeasible on a 256 x 256 grid.  For a uniform grid,
its entries depend only on the relative displacement of two cells, so the
matrix-vector product is an exact linear convolution.  This module evaluates
that convolution with a zero-padded FFT and never materializes the dense
matrix.  The electromagnetic parameters are deliberately dimensionless: the
UrbanRadio3D files used by this project do not expose a verified carrier
frequency or physical pixel spacing.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence

import numpy as np
from scipy.special import hankel2, j1
import torch
from torch import Tensor, nn
from torch.nn import functional as F


GridSize = int | tuple[int, int] | Sequence[int]


def _grid_shape(grid_size: GridSize) -> tuple[int, int]:
    if isinstance(grid_size, int):
        shape = (grid_size, grid_size)
    else:
        values = tuple(int(value) for value in grid_size)
        if len(values) != 2:
            raise ValueError(f"grid_size must contain two values, got {values}")
        shape = values
    if min(shape) <= 0:
        raise ValueError(f"grid dimensions must be positive, got {shape}")
    return shape


def _validate_physics(dimensionless_wavenumber: float, grid_spacing: float) -> None:
    if not math.isfinite(dimensionless_wavenumber) or dimensionless_wavenumber <= 0:
        raise ValueError("dimensionless_wavenumber must be finite and positive")
    if not math.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be finite and positive")


def building_contrast(
    building_mask: np.ndarray,
    *,
    epsilon_air: float = 1.0,
    epsilon_building: float = 4.0,
    conductivity: float = 0.0,
) -> np.ndarray:
    """Return complex contrast ``chi`` for a binary building footprint.

    ``conductivity`` is the dimensionless ratio ``sigma / (omega epsilon_0)``.
    It defaults to zero because the current dataset contains no material or
    carrier metadata.  It must therefore be reported as an adaptation rather
    than interpreted as a measured material property.
    """

    if not math.isfinite(epsilon_air) or epsilon_air <= 0:
        raise ValueError("epsilon_air must be finite and positive")
    if not math.isfinite(epsilon_building) or epsilon_building <= 0:
        raise ValueError("epsilon_building must be finite and positive")
    if not math.isfinite(conductivity) or conductivity < 0:
        raise ValueError("conductivity must be finite and non-negative")
    mask = np.asarray(building_mask, dtype=np.bool_)
    value = complex(epsilon_building / epsilon_air - 1.0, -conductivity)
    return np.where(mask, value, 0.0j).astype(np.complex64, copy=False)


@lru_cache(maxsize=32)
def _green_offset_lookup(
    height: int,
    width: int,
    dimensionless_wavenumber: float,
    grid_spacing: float,
) -> np.ndarray:
    """Cache the free-space Green function for every possible grid offset."""

    _validate_physics(dimensionless_wavenumber, grid_spacing)
    row_offsets = np.arange(-(height - 1), height, dtype=np.float64)
    col_offsets = np.arange(-(width - 1), width, dtype=np.float64)
    dr, dc = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    distance = np.hypot(dr, dc) * grid_spacing

    # The point-source Green function is singular at the source cell.  The
    # paper states that E_inc is averaged over a cell but does not give the
    # closed form used in its code.  The equal-area cell radius is a finite,
    # deterministic approximation and is recorded in the adaptation docs.
    cell_radius = grid_spacing / math.sqrt(math.pi)
    effective_distance = np.where(distance == 0.0, cell_radius, distance)
    values = -0.25j * hankel2(
        0, dimensionless_wavenumber * effective_distance
    )
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise FloatingPointError("non-finite values in incident-field Green kernel")
    result = values.astype(np.complex64)
    result.setflags(write=False)
    return result


def build_incident_field(
    grid_size: GridSize,
    tx_row_col: tuple[int, int] | Sequence[int],
    *,
    dimensionless_wavenumber: float,
    grid_spacing: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Build a leakage-free two-channel incident field from Tx geometry only.

    The return value has shape ``[2, H, W]`` in real/imaginary channel order.
    A shared complex-amplitude normalization preserves phase and the linear VIE
    relationship; no path-loss labels or split-wide statistics are used.
    """

    height, width = _grid_shape(grid_size)
    values = tuple(int(value) for value in tx_row_col)
    if len(values) != 2:
        raise ValueError("tx_row_col must contain exactly two integers")
    tx_row, tx_col = values
    if not (0 <= tx_row < height and 0 <= tx_col < width):
        raise ValueError(
            f"Tx raster coordinate {(tx_row, tx_col)} outside {(height, width)}"
        )
    lookup = _green_offset_lookup(
        height,
        width,
        float(dimensionless_wavenumber),
        float(grid_spacing),
    )
    row_start = (height - 1) - tx_row
    col_start = (width - 1) - tx_col
    field = np.asarray(
        lookup[row_start : row_start + height, col_start : col_start + width]
    ).copy()
    if normalize:
        scale = float(np.abs(field).max(initial=0.0))
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError("incident field has invalid normalization scale")
        field /= scale
    return np.stack((field.real, field.imag), axis=0).astype(np.float32, copy=False)


def _vie_kernel(
    height: int,
    width: int,
    dimensionless_wavenumber: float,
    grid_spacing: float,
) -> np.ndarray:
    """Construct the relative-offset kernel from PEFNet Eqs. (11)-(12)."""

    _validate_physics(dimensionless_wavenumber, grid_spacing)
    row_offsets = np.arange(-(height - 1), height, dtype=np.float64)
    col_offsets = np.arange(-(width - 1), width, dtype=np.float64)
    dr, dc = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    distance = np.hypot(dr, dc) * grid_spacing
    area = grid_spacing**2
    radius = math.sqrt(area / math.pi)
    ka = dimensionless_wavenumber * radius

    safe_distance = np.where(distance == 0.0, radius, distance)
    kernel = (
        0.5j
        * math.pi
        * dimensionless_wavenumber
        * radius
        * j1(ka)
        * hankel2(0, dimensionless_wavenumber * safe_distance)
    )
    self_term = 0.5j * (math.pi * ka * hankel2(1, ka) - 2.0j)
    kernel[height - 1, width - 1] = self_term
    if not np.isfinite(kernel.real).all() or not np.isfinite(kernel.imag).all():
        raise FloatingPointError("non-finite values in VIE convolution kernel")
    return kernel.astype(np.complex64, copy=False)


def _to_batched_complex(value: Tensor, name: str) -> Tensor:
    """Accept complex maps or real/imag channel tensors and return [B,H,W]."""

    if value.is_complex():
        if value.ndim == 2:
            return value.unsqueeze(0)
        if value.ndim == 3:
            return value
        raise ValueError(f"{name} complex tensor must be [H,W] or [B,H,W]")
    if value.ndim == 2:
        return torch.complex(value.float(), torch.zeros_like(value, dtype=torch.float32)).unsqueeze(0)
    if value.ndim == 3:
        if value.shape[0] == 2:
            value = value.unsqueeze(0)
        else:
            return torch.complex(value.float(), torch.zeros_like(value, dtype=torch.float32))
    if value.ndim != 4 or value.shape[1] not in (1, 2):
        raise ValueError(
            f"{name} must be complex [B,H,W] or real [B,1|2,H,W], got {tuple(value.shape)}"
        )
    real = value[:, 0].float()
    imaginary = value[:, 1].float() if value.shape[1] == 2 else torch.zeros_like(real)
    return torch.complex(real, imaginary)


def _resize_complex(value: Tensor, size: tuple[int, int], *, mode: str) -> Tensor:
    if value.shape[-2:] == size:
        return value
    channels = torch.stack((value.real, value.imag), dim=1)
    if mode == "nearest":
        resized = F.interpolate(channels, size=size, mode=mode)
    else:
        resized = F.interpolate(channels, size=size, mode=mode, align_corners=False)
    return torch.complex(resized[:, 0], resized[:, 1])


class FFTVIEOperator(nn.Module):
    """Evaluate a fixed-grid VIE residual with differentiable FFT convolution."""

    def __init__(
        self,
        grid_size: GridSize,
        dimensionless_wavenumber: float,
        *,
        grid_spacing: float = 1.0,
        epsilon_air: float = 1.0,
        epsilon_building: float = 4.0,
        conductivity: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = _grid_shape(grid_size)
        self.dimensionless_wavenumber = float(dimensionless_wavenumber)
        self.grid_spacing = float(grid_spacing)
        self.epsilon_air = float(epsilon_air)
        self.epsilon_building = float(epsilon_building)
        self.conductivity = float(conductivity)
        _validate_physics(self.dimensionless_wavenumber, self.grid_spacing)
        # Validate the documented material assumptions even though the batch
        # supplies contrast explicitly.  Keeping them on the operator makes
        # provenance serialization straightforward for training scripts.
        building_contrast(
            np.zeros((1, 1), dtype=np.bool_),
            epsilon_air=self.epsilon_air,
            epsilon_building=self.epsilon_building,
            conductivity=self.conductivity,
        )

        height, width = self.grid_size
        kernel = torch.from_numpy(
            _vie_kernel(
                height,
                width,
                self.dimensionless_wavenumber,
                self.grid_spacing,
            )
        )
        self.fft_shape = (3 * height - 2, 3 * width - 2)
        kernel_fft = torch.fft.fft2(kernel, s=self.fft_shape)
        self.register_buffer("_kernel_fft", kernel_fft, persistent=False)

    def _prepare(
        self,
        total_field: Tensor,
        incident_field: Tensor,
        contrast: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        total = _resize_complex(
            _to_batched_complex(total_field, "total_field"),
            self.grid_size,
            mode="bilinear",
        )
        incident = _resize_complex(
            _to_batched_complex(incident_field, "incident_field"),
            self.grid_size,
            mode="bilinear",
        )
        chi = _resize_complex(
            _to_batched_complex(contrast, "contrast"),
            self.grid_size,
            mode="nearest",
        )
        batch = total.shape[0]
        prepared = []
        for name, value in (("incident_field", incident), ("contrast", chi)):
            if value.shape[0] == 1 and batch > 1:
                value = value.expand(batch, -1, -1)
            if value.shape[0] != batch:
                raise ValueError(
                    f"{name} batch {value.shape[0]} does not match total_field batch {batch}"
                )
            prepared.append(value)
        return total, prepared[0], prepared[1]

    def residual(
        self,
        total_field: Tensor,
        incident_field: Tensor,
        contrast: Tensor,
    ) -> Tensor:
        """Return real/imaginary VIE residual channels with shape [B,2,h,w]."""

        total, incident, chi = self._prepare(total_field, incident_field, contrast)
        source = chi * total
        source_fft = torch.fft.fft2(source, s=self.fft_shape)
        kernel_fft = self._kernel_fft.to(device=source.device, dtype=source.dtype)
        full = torch.fft.ifft2(source_fft * kernel_fft, s=self.fft_shape)
        height, width = self.grid_size
        scattered = full[
            ...,
            height - 1 : height - 1 + height,
            width - 1 : width - 1 + width,
        ]
        value = total + scattered - incident
        return torch.stack((value.real, value.imag), dim=1)

    def loss(
        self,
        total_field: Tensor,
        incident_field: Tensor,
        contrast: Tensor,
        *,
        reduction: str = "mean",
        epsilon: float = 1e-12,
    ) -> Tensor:
        """Return grid-size-invariant RMS magnitude of the VIE residual.

        The source paper writes ``||r||_2 / N``.  This adaptation evaluates
        physics on a configurable coarse grid (64 x 64 in the formal config),
        so RMS normalization avoids changing the effective physics weight when
        that computational grid size changes.  The deviation is intentional
        and must be recorded in experiment provenance.
        """

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        residual = self.residual(total_field, incident_field, contrast)
        squared_magnitude = residual.square().sum(dim=1)
        per_sample = torch.sqrt(
            squared_magnitude.mean(dim=(-2, -1)) + float(epsilon)
        )
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()

    residual_loss = loss

    def forward(
        self,
        total_field: Tensor,
        incident_field: Tensor,
        contrast: Tensor,
    ) -> Tensor:
        return self.residual(total_field, incident_field, contrast)


# Concise public name requested by the training/evaluation integration.
VIEOperator = FFTVIEOperator


__all__ = [
    "FFTVIEOperator",
    "VIEOperator",
    "build_incident_field",
    "building_contrast",
]
