import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from models import RadioDiff3D


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition(batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    """Build only the conditions declared by the RadioDiff-3D protocol.

    Full psi, LoS, and AFF depth are deliberately excluded. The paper's
    sampling configuration adds one sparse-RM channel, but no dense target or
    unobserved target values.
    """
    channels = [batch["q"], batch["building_mask"], batch["building_height"]]
    if bool(cfg["data"].get("use_sampling", False)):
        channels.append(batch["s"])
    return torch.cat(channels, dim=1)


def build_model(cfg: Dict[str, Any]) -> RadioDiff3D:
    return RadioDiff3D(**cfg["model"])


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
