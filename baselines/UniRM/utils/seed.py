"""Reproducibility helpers."""

from __future__ import annotations

import random



def seed_python(seed: int) -> None:
    """Seed Python's standard pseudo-random generator."""

    random.seed(seed)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA generators."""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
