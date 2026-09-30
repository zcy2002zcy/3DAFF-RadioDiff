"""Regression metrics evaluated on the original raw target scale."""

from __future__ import annotations

import torch
from torch import Tensor


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    error = prediction - target
    mse = torch.mean(error.square())
    denominator = torch.mean(target.square()).clamp_min(1e-12)
    return {
        "rmse": torch.sqrt(mse).item(),
        "nmse": (mse / denominator).item(),
        "mae": torch.mean(error.abs()).item(),
    }
