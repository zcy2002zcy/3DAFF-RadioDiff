"""Metrics aligned with the 3D-RadioDiff evaluation implementation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


METRIC_NAMES = ("nmse", "rmse", "ssim", "psnr")


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def _gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, window_size, window_size).expand(
        channels, 1, window_size, window_size
    ).contiguous()


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

    channels = pred.shape[1]
    window = _gaussian_window(channels, pred.device, pred.dtype, window_size, sigma)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c3 = c2 / 2.0
    luminance = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1 + eps)
    sigma_x = torch.sqrt(torch.clamp(sigma_x2, min=0.0) + eps)
    sigma_y = torch.sqrt(torch.clamp(sigma_y2, min=0.0) + eps)
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x2 + sigma_y2 + c2 + eps)
    structure = (sigma_xy + c3) / (sigma_x * sigma_y + c3 + eps)
    return (luminance * contrast * structure).mean(dim=(1, 2, 3))


def aligned_batch_metrics(
    prediction_zero_one: torch.Tensor,
    target_zero_one: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-sample metrics using the main model's [-1, 1] convention."""
    prediction = prediction_zero_one.float().clamp(0.0, 1.0) * 2.0 - 1.0
    target = target_zero_one.float().clamp(0.0, 1.0) * 2.0 - 1.0
    squared_error = torch.sum((prediction - target) ** 2, dim=(1, 2, 3))
    target_energy = torch.sum(target**2, dim=(1, 2, 3))
    mse = _mse(prediction, target)
    return {
        "nmse": squared_error / (target_energy + 1e-8),
        "rmse": torch.sqrt(mse + 1e-12),
        "ssim": _ssim(prediction, target, data_range=2.0),
        "psnr": 10.0 * torch.log10(4.0 / (mse + 1e-12)),
    }
