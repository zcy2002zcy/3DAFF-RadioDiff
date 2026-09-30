"""Evaluation visualizations with invalid Rx locations explicitly masked."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_reconstruction_figure(
    path: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Save GT, prediction, and absolute error with one shared target scale."""

    path.parent.mkdir(parents=True, exist_ok=True)
    gt = np.where(valid_mask, ground_truth, np.nan)
    pred = np.where(valid_mask, prediction, np.nan)
    error = np.where(valid_mask, np.abs(prediction - ground_truth), np.nan)
    valid_values = ground_truth[valid_mask]
    vmin = float(valid_values.min())
    vmax = float(valid_values.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    first = axes[0].imshow(gt, origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Ground Truth (raw)")
    axes[1].imshow(pred, origin="upper", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Prediction (raw)")
    third = axes[2].imshow(error, origin="upper", vmin=0, cmap="magma")
    axes[2].set_title("Absolute Error")
    for axis in axes:
        axis.set_xlabel("Y / column")
        axis.set_ylabel("row")
    fig.colorbar(first, ax=axes[:2], shrink=0.8)
    fig.colorbar(third, ax=axes[2], shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)

