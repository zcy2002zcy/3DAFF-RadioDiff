"""Auditable no-sampling adapter for the pinned official UniRM Prompt-UNet.

The upstream network expects 128x128 SpectrumNet tensors whose second channel
contains sparse ground-truth RSS.  That would violate this project's
sampling-free protocol.  This adapter preserves the official network and
prompt modules, replaces sparse RSS with a Tx indicator, and removes the
upstream hard-coded identity addition of that second channel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "23d03510b7ae7ef1eeae4195269e2ebf86de0341"


def _load_official_module():
    root = Path(__file__).resolve().parents[1] / "external" / "UniRM"
    source = root / "cond_unet.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Pinned UniRM source missing: {source}. Clone the official repository first."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("pinned_official_unirm", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import UniRM from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_args(enable_prompt: bool) -> SimpleNamespace:
    return SimpleNamespace(
        size="1",
        enable_prompt=int(enable_prompt),
        prompt_content="f_h_b",
        num_memory_spatial=256,
        in_channels=3,
        conv_num=2,
        dilations=[1, 4, 8],
        num_memory_freq=256,
        num_memory_height=256,
    )


class UniRMAdapted(nn.Module):
    """Official UniRM size-1 model adapted to zero-measurement 256x256 maps."""

    def __init__(self, *, enable_prompt: bool) -> None:
        super().__init__()
        self.enable_prompt = bool(enable_prompt)
        official = _load_official_module()
        self.network = official.PromptUnet_model(args=_official_args(enable_prompt))

    @staticmethod
    def _distance_prior(tx: Tensor) -> Tensor:
        batch, _, height, width = tx.shape
        flat_index = tx.flatten(2).argmax(dim=-1)
        tx_row = torch.div(flat_index, width, rounding_mode="floor").to(tx.dtype)
        tx_col = (flat_index % width).to(tx.dtype)
        rows = torch.arange(height, device=tx.device, dtype=tx.dtype).view(1, height, 1)
        cols = torch.arange(width, device=tx.device, dtype=tx.dtype).view(1, 1, width)
        distance = torch.sqrt(
            (rows - tx_row.view(batch, 1, 1)).square()
            + (cols - tx_col.view(batch, 1, 1)).square()
        )
        maximum = max(float((height - 1) ** 2 + (width - 1) ** 2) ** 0.5, 1.0)
        return (1.0 - distance / maximum).clamp(0.0, 1.0).unsqueeze(1)

    def _official_input(self, dense_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if dense_input.ndim != 4 or dense_input.shape[1] != 3:
            raise ValueError("UniRM adapter expects [building height, Tx mask, Rx height]")
        building = dense_input[:, 0:1].clamp(0.0, 1.0)
        tx = dense_input[:, 1:2]
        rx_height = dense_input[:, 2:3]
        building_small = F.interpolate(building, size=(128, 128), mode="bilinear", align_corners=False)
        tx_small = F.adaptive_max_pool2d(tx, (128, 128))
        rx_small = F.interpolate(rx_height, size=(128, 128), mode="nearest")
        building_m = building_small * 255.0
        rx_m = rx_small * 20.0
        current_occupancy = (building_m >= rx_m).to(building_small.dtype)
        occupancy_1m = (building_m >= 1.0).to(building_small.dtype)
        occupancy_2m = (building_m >= 2.0).to(building_small.dtype)
        occupancy_4m = (building_m >= 4.0).to(building_small.dtype)
        distance_prior = self._distance_prior(tx_small)
        official_input = torch.cat(
            (
                current_occupancy,
                tx_small,
                occupancy_1m,
                occupancy_2m,
                occupancy_4m,
                distance_prior,
                rx_small,
            ),
            dim=1,
        )
        height_value = rx_height[:, 0, 0, 0] * 20.0
        frequency_value = torch.zeros_like(height_value)
        return official_input, height_value, frequency_value

    def forward(self, dense_input: Tensor) -> Tensor:
        official_input, height_value, frequency_value = self._official_input(dense_input)
        # Upstream returns input[:, 1:2] + residual because that channel is sparse
        # RSS.  Here it is a Tx mask, so subtract it to retain only the prediction.
        prediction_small = self.network(
            official_input, frequency_value, height_value
        ) - official_input[:, 1:2]
        return F.interpolate(
            prediction_small,
            size=dense_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def build_unirm_adapted(*, enable_prompt: bool) -> UniRMAdapted:
    return UniRMAdapted(enable_prompt=enable_prompt)

