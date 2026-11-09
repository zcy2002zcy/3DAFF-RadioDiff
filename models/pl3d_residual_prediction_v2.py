"""Isolated residual-diffusion v2 math; inference uses only known conditions.

The existing architecture, cosine schedule and valid-area metrics are unchanged.
Geometry projection fixes the *clean residual* inside buildings to zero, not the
radio-map prediction. No target or evaluator mask is an inference argument.
"""
from __future__ import annotations

import math

import torch

from models.pl3d_residual_prediction import (
    Corrector, Predictor, gradient_loss, schedule, x0_from_v,
)


def available_from_condition(condition):
    """Recover outdoor voxels from supplied normalized building heights only.

    This helper is deliberately specific to aff_prediction at heights 1--4 m.
    Channel 1 is clip(building_height_m / 20, 0, 1); clipping above 20 m does
    not alter the availability predicate at these receiver heights.
    """
    if condition.ndim != 5 or condition.shape[1] != 9 or condition.shape[2] != 4:
        raise ValueError('Expected aff_prediction condition [B,9,4,H,W]')
    if not condition.is_floating_point():
        raise TypeError('Condition must contain floating-point normalized heights')
    receiver_heights = condition.new_tensor([1., 2., 3., 4.]).view(1, 1, 4, 1, 1)
    return condition[:, 1:2] < receiver_heights / 20.


def draw_timesteps(batch, alpha, device, terminal_probability=.25):
    """Uniform timestep draw with an explicit terminal-state mixture.

    P(t=T-1) = p + (1-p)/T; other states each have probability (1-p)/T.
    This is a changed training objective (not importance-corrected uniform DDPM).
    Global torch RNG state is used so normal checkpoint RNG restoration applies.
    """
    if alpha.ndim != 1 or len(alpha) < 1:
        raise ValueError('alpha must be a nonempty one-dimensional schedule')
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be a positive integer')
    if not 0. <= terminal_probability <= 1.:
        raise ValueError('terminal_probability must be within [0,1]')
    t = torch.randint(0, len(alpha), (batch,), device=device)
    if terminal_probability > 0.:
        terminal = torch.rand(batch, device=device) < terminal_probability
        t = torch.where(terminal, torch.full_like(t, len(alpha)-1), t)
    return t


def _masked_mean(value, mask):
    """Differentiable zero for empty masks; no host/device synchronization."""
    mask = mask.bool()
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def diffusion_losses(v, xt, v_target, alpha_batch, coarse, target, valid, scale,
                     invalid_weight=.1, high_noise_weight=1., high_noise_threshold=.2):
    """Legacy losses plus geometry v supervision and high-noise clean-residual MSE.

    Inputs are FP32 computation tensors (casts here also protect AMP callers).
    `v_target` must be formed using a zero clean residual at invalid voxels.
    High-noise MSE is computed in normalized *residual* units (map MSE / scale²)
    over valid voxels whose cumulative alpha is at most the specified threshold.
    Components are unweighted scalar tensors for transparent progress logs.
    """
    if not math.isfinite(float(scale)) or scale <= 0.:
        raise ValueError('scale must be finite and positive')
    if invalid_weight < 0. or high_noise_weight < 0.:
        raise ValueError('Loss weights must be nonnegative')
    if not 0. <= high_noise_threshold <= 1.:
        raise ValueError('high_noise_threshold must be within [0,1]')
    if not (v.shape == xt.shape == v_target.shape == coarse.shape == target.shape == valid.shape):
        raise ValueError('Prediction, target and mask shapes must match')
    if v.ndim != 5 or v.shape[1] != 1:
        raise ValueError('Expected residual volumes [B,1,D,H,W]')
    if alpha_batch.numel() != v.shape[0]:
        raise ValueError('alpha_batch must contain one value per example')
    a = alpha_batch.float().reshape(-1, 1, 1, 1, 1)
    valid = valid.bool()
    clean = x0_from_v(xt.float(), v.float(), a)
    prediction = coarse.float() + scale * clean
    map_error = prediction - target.float()
    v_error = (v.float() - v_target.float()).square()
    components = {
        'valid_v_mse': _masked_mean(v_error, valid),
        'map_l1': _masked_mean(map_error.abs(), valid),
        'gradient_l1': gradient_loss(prediction, target.float(), valid),
        'invalid_v_mse': _masked_mean(v_error, ~valid),
        'high_noise_residual_mse': _masked_mean(
            map_error.square() / (scale * scale), valid & (a <= high_noise_threshold)),
    }
    total = (components['valid_v_mse'] + .5 * components['map_l1']
             + .05 * components['gradient_l1']
             + invalid_weight * components['invalid_v_mse']
             + high_noise_weight * components['high_noise_residual_mse'])
    return total, components


@torch.no_grad()
def sample_residual(corrector, condition, coarse, alpha, scale, steps=30, seed=42,
                    project_geometry=True):
    """Return an unclipped map-space delta using Gaussian-initialized DDIM eta=0.

    At each transition the known building-interior clean residual is zero and
    its noisy state is sqrt(1-alpha_next) times the fixed initial Gaussian.
    Outdoor DDIM updates are unchanged. The final interior delta is exactly zero.
    """
    if alpha.ndim != 1 or len(alpha) < 1 or not 1 <= steps <= len(alpha):
        raise ValueError('Sampling steps must be between 1 and diffusion timesteps')
    if not math.isfinite(float(scale)) or scale <= 0.:
        raise ValueError('scale must be finite and positive')
    available = available_from_condition(condition)
    if coarse.shape != available.shape:
        raise ValueError('Coarse map must have shape [B,1,4,H,W] matching condition')
    if alpha.device != condition.device or coarse.device != condition.device:
        raise ValueError('Schedule, conditions and coarse map must share a device')
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    initial_noise = torch.randn(coarse.shape, device=condition.device,
                                dtype=coarse.dtype, generator=generator)
    x = initial_noise
    indices = torch.linspace(len(alpha)-1, 0, steps, device=condition.device).round().long()
    for i, t in enumerate(indices):
        a = alpha[t]
        v = corrector(x, condition, t.expand(x.shape[0]), coarse).float()
        if v.shape != x.shape:
            raise ValueError('Corrector output must match noisy residual shape')
        clean = x0_from_v(x, v, a)
        eps = (1-a).sqrt()*x + a.sqrt()*v
        previous = alpha[indices[i+1]] if i+1 < len(indices) else x.new_tensor(1.)
        if project_geometry:
            clean = torch.where(available, clean, torch.zeros_like(clean))
        x = previous.sqrt()*clean + (1-previous).sqrt()*eps
        if project_geometry:
            x = torch.where(available, x, (1-previous).sqrt()*initial_noise)
    return scale*x


@torch.no_grad()
def predict(corrector, condition, coarse, alpha, scale, steps=30, seed=42,
            project_geometry=True, strength=1.):
    """Clipped map prediction; strength is a validation-locked global scalar."""
    if not math.isfinite(float(strength)) or not 0. <= strength <= 1.:
        raise ValueError('strength must be finite and within [0,1]')
    if strength == 0.:
        return coarse.clamp(-1, 1)
    delta = sample_residual(corrector, condition, coarse, alpha, scale,
                            steps=steps, seed=seed, project_geometry=project_geometry)
    return (coarse + strength*delta).clamp(-1, 1)
