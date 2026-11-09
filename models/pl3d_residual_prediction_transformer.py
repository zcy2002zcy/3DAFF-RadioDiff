"""Strict prediction v3: the original spatial Transformer in both 3-D stages.

The inherited v2 CNN, AFF, input selection and timestep embedding are unchanged.
At each stage, its two existing bottleneck residual blocks surround a new stack
of two original-configuration Transformer blocks. Four height planes share that
stage's stack; attention is spatial within a plane, never across height planes.
The 3-D convolutions before/after the stack still mix information across height.

The attention weights/FFN are those of ``models.transformer.TransformerBlock``.
Only ``need_weights=False`` changes its attention execution. This computes the
same mathematical operation, not a promise of bit-for-bit equality across
attention kernels, devices or floating-point dtypes. Optional training-only
activation recomputation and serial height execution also preserve the graph's
mathematics. No position encoding, window or channel projection is introduced.

This file can be loaded privately with importlib; the original project's
``models`` package must already be importable. No legacy source is patched.
"""
from __future__ import annotations

import torch
from torch import nn
from compat_runtime import activation_checkpoint

from models.pl3d_residual_prediction import (
    Predictor as _OriginalPredictor,
    Corrector as _OriginalCorrector,
)
from models.radiodiff_aff_full3d import ResidualBlock3D
from models.transformer import TransformerBlock as _OriginalTransformerBlock


ARCHITECTURE_SPEC = {
    "id": "pl3d_aff_prediction_transformer_v3",
    "base_channels": 32,
    "cnn_channels": [32, 64, 128],
    "receiver_heights_m": [1, 2, 3, 4],
    "condition_schema_channels": 9,
    "allowed_condition_indices": [0, 1, 2, 5, 6],
    "ignored_condition_indices": [3, 4, 7, 8],
    "condition_profile": "aff_prediction",
    "sample_rate": 0.0,
    "coarse_effective_input_channels": 5,
    "corrector_effective_input_channels": 6,
    "insertion": "ResidualBlock3D -> PerHeightSpatialTransformer -> ResidualBlock3D",
    "transformer": {
        "dim": 128,
        "heads": 4,
        "ffn_dim": 256,
        "depth": 2,
        "dropout": 0.0,
        "activation": "gelu",
        "normalization": "pre_layernorm",
        "layernorm_eps": 1e-5,
        "attention_bias": True,
        "ffn_bias": True,
        "residual_connections": "attention_and_ffn",
        "attention_scope": "H*W spatial tokens within each height",
        "cross_height_attention": False,
        "height_parameter_sharing": True,
        "cross_stage_parameter_sharing": False,
        "position_encoding": None,
        "window": None,
        "channel_projection": None,
        "need_weights": False,
        "height_execution": "serial",
        "activation_checkpoint": "training_only_API_selected_recompute_preserve_rng",
        "equivalence": "mathematically equivalent; floating-point kernels may differ",
        "parameters_per_stage": 264960,
        "parameters_both_stages": 529920,
    },
    "legacy_checkpoint_compatible": False,
    "transformer_init_rng": "CPU fork_rng; preserve global RNG after inherited CNN init",
}


class OriginalConfigTransformerBlock(_OriginalTransformerBlock):
    """Original block and state-dict layout, with unreturned attention weights.

    Hyperparameters are deliberately not configurable: this experiment tests
    exactly the original 128/4/256/2 stack, not a separately tuned Transformer.
    """

    def __init__(self) -> None:
        super().__init__(dim=128, num_heads=4, mlp_dim=256,
                         dropout=0.0, activation="gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 128:
            raise ValueError("Transformer input must be [B,128,H,W]")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError("Transformer batch and spatial dimensions must be positive")
        b, c, h, w = x.shape
        sequence = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(sequence)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.norm2(sequence))
        return sequence.transpose(1, 2).reshape(b, c, h, w)


class PerHeightSpatialTransformer(nn.Module):
    """One shared two-block spatial stack, applied separately to four heights."""

    def __init__(self, *, activation_checkpoint: bool = True) -> None:
        super().__init__()
        if not isinstance(activation_checkpoint, bool):
            raise TypeError("activation_checkpoint must be a bool")
        self.activation_checkpoint = activation_checkpoint
        self.blocks = nn.ModuleList([OriginalConfigTransformerBlock() for _ in range(2)])

    def _forward_height(self, plane: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            plane = block(plane)
        return plane

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5 or volume.shape[1] != 128 or volume.shape[2] != 4:
            raise ValueError("Per-height Transformer input must be [B,128,4,H,W]")
        if volume.shape[0] < 1 or min(volume.shape[-2:]) < 1:
            raise ValueError("Transformer batch and spatial dimensions must be positive")
        planes = []
        for height in range(4):
            plane = volume[:, :, height]
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                plane = activation_checkpoint(self._forward_height, plane)
            else:
                plane = self._forward_height(plane)
            planes.append(plane)
        return torch.stack(planes, dim=2)


def _check_base_channels(base_channels: int) -> None:
    if isinstance(base_channels, bool) or base_channels != 32:
        raise ValueError("This matched-configuration experiment requires base_channels=32")


def _install_transformer(net: nn.Module, activation_checkpoint: bool) -> None:
    if len(net.bottleneck) != 2 or not all(
        isinstance(block, ResidualBlock3D) for block in net.bottleneck
    ):
        raise RuntimeError("Inherited bottleneck changed; expected two ResidualBlock3D modules")
    first, second = net.bottleneck
    # Constructors run on CPU, before the caller's .to(device). New parameters
    # must not perturb subsequent loader/noise RNG or the other inherited stage.
    with torch.random.fork_rng(devices=[]):
        spatial = PerHeightSpatialTransformer(activation_checkpoint=activation_checkpoint)
    net.bottleneck = nn.Sequential(first, spatial, second)


def _validate_condition(condition: torch.Tensor) -> None:
    if condition.ndim != 5 or condition.shape[1] != 9 or condition.shape[2] != 4:
        raise ValueError("condition must be [B,9,4,H,W]")
    if condition.shape[0] < 1 or min(condition.shape[-2:]) < 4:
        raise ValueError("Condition needs B>=1 and H,W>=4 for two CNN downsampling blocks")
    if not condition.is_floating_point():
        raise TypeError("condition must be a floating-point tensor")


def _validate_signal(name: str, signal: torch.Tensor, condition: torch.Tensor) -> None:
    expected = (condition.shape[0], 1, *condition.shape[2:])
    if tuple(signal.shape) != expected or signal.device != condition.device:
        raise ValueError(f"{name} must have shape {expected} on the condition device")
    if not signal.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


class Predictor(_OriginalPredictor):
    """Deterministic coarse prediction, with original-config spatial attention."""

    model_type = "pl3d_aff_prediction_transformer_v3_coarse"

    def __init__(self, base_channels: int = 32, *, activation_checkpoint: bool = True) -> None:
        _check_base_channels(base_channels)
        super().__init__(base_channels=base_channels)
        _install_transformer(self, activation_checkpoint)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        _validate_condition(condition)
        return super().forward(condition)


class Corrector(_OriginalCorrector):
    """Residual v-prediction; no target, measurements or validity-mask argument."""

    model_type = "pl3d_aff_prediction_transformer_v3_corrector"

    def __init__(self, base_channels: int = 32, *, activation_checkpoint: bool = True) -> None:
        _check_base_channels(base_channels)
        super().__init__(base_channels=base_channels)
        _install_transformer(self, activation_checkpoint)

    def forward(self, noisy_residual: torch.Tensor, condition: torch.Tensor,
                t: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        _validate_condition(condition)
        _validate_signal("noisy_residual", noisy_residual, condition)
        _validate_signal("coarse", coarse, condition)
        if t.ndim != 1 or t.shape[0] != condition.shape[0] or t.device != condition.device:
            raise ValueError("t must be [B] on the condition device")
        if t.dtype not in (torch.int32, torch.int64):
            raise TypeError("t must contain integer diffusion timestep indices")
        return super().forward(noisy_residual, condition, t, coarse)


__all__ = ["ARCHITECTURE_SPEC", "OriginalConfigTransformerBlock",
           "PerHeightSpatialTransformer", "Predictor", "Corrector"]
