"""Deterministic IDW coarse prior computed only from sparse observations."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def make_idw_prior(
    measurement: np.ndarray,
    measurement_mask: np.ndarray,
    *,
    neighbors: int = 12,
    power: float = 2.0,
) -> np.ndarray:
    """Return [D,H,W] IDW maps while preserving every measured value exactly."""
    if measurement.shape != measurement_mask.shape or measurement.ndim != 3:
        raise ValueError("measurement and mask must share [D,H,W]")
    depth, height, width = measurement.shape
    grid = np.stack(
        np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    output = np.empty_like(measurement, dtype=np.float32)
    for z in range(depth):
        known = measurement_mask[z] > 0
        known_xy = np.argwhere(known)
        if len(known_xy) == 0:
            raise ValueError(f"height index {z} has no sparse observations")
        values = measurement[z][known].astype(np.float64, copy=False)
        k = min(neighbors, len(known_xy))
        distances, indices = cKDTree(known_xy).query(grid, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-8) ** power
        plane = (
            np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
        ).reshape(height, width)
        plane[known] = measurement[z][known]
        output[z] = np.clip(plane, -1.0, 1.0).astype(np.float32)
    return output
