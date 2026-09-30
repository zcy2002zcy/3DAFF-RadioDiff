"""Shared, leakage-safe tensor operations for all sampled PL methods."""

from __future__ import annotations

import torch


def validate_target_and_mask(target: torch.Tensor, sample_mask: torch.Tensor) -> None:
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"target must be [B,1,H,W], got {tuple(target.shape)}")
    if sample_mask.shape != target.shape:
        raise ValueError(f"mask/target mismatch: {sample_mask.shape} vs {target.shape}")
    if not torch.isfinite(target).all():
        raise ValueError("target contains NaN or Inf")
    binary = (sample_mask == 0) | (sample_mask == 1)
    if not bool(binary.all()):
        raise ValueError("sample_mask must be binary")


def make_sparse_observation(
    target: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """Expose target values only at authorized sampled pixels."""
    validate_target_and_mask(target, sample_mask)
    return torch.where(sample_mask.bool(), target, torch.zeros_like(target))


def assert_sparse_is_leakage_safe(
    sparse: torch.Tensor,
    target: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    atol: float = 0.0,
) -> None:
    """Fail if sparse input contains any unobserved target value."""
    validate_target_and_mask(target, sample_mask)
    if sparse.shape != target.shape:
        raise ValueError(f"sparse/target mismatch: {sparse.shape} vs {target.shape}")
    zeros = torch.zeros_like(sparse)
    if not torch.allclose(
        torch.where(sample_mask.bool(), zeros, sparse), zeros, atol=atol, rtol=0.0
    ):
        raise AssertionError("non-zero value found outside the observation mask")
    if not torch.allclose(
        sparse[sample_mask.bool()], target[sample_mask.bool()], atol=atol, rtol=0.0
    ):
        raise AssertionError("sampled values do not equal target values")


def validate_prediction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """Check only interface properties; never tune output using the target."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains NaN or Inf")


def apply_final_measurement_projection(
    prediction: torch.Tensor,
    sparse: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the same final hard observation constraint to every sampled method."""
    if prediction.shape != sparse.shape or prediction.shape != sample_mask.shape:
        raise ValueError("prediction, sparse, and sample_mask must have identical shapes")
    return torch.where(sample_mask.bool(), sparse, prediction)


def full_slice_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return per-slice NMSE/RMSE/PSNR; macro-average outside this function."""
    validate_prediction(prediction, target)
    dims = (1, 2, 3)
    squared_error = (prediction.float() - target.float()).square()
    mse = squared_error.mean(dim=dims)
    nmse = squared_error.sum(dim=dims) / target.float().square().sum(dim=dims).clamp_min(eps)
    return {
        "nmse": nmse,
        "rmse": torch.sqrt(mse.clamp_min(0.0)),
        "psnr": 10.0 * torch.log10((data_range**2) / mse.clamp_min(eps)),
    }
