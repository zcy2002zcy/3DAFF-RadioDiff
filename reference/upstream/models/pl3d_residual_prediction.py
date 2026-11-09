"""Two-stage strict prediction. No measurements, masks or target enter inference."""
from __future__ import annotations

import math
import torch
from torch import nn
from models.radiodiff_aff_full3d import RadioDiffAFFPrediction3DUNet, ResidualBlock3D


class Predictor(RadioDiffAFFPrediction3DUNet):
    def forward(self, condition):
        blank = torch.zeros_like(condition[:, :1])
        t = torch.zeros(condition.shape[0], dtype=torch.long, device=condition.device)
        return super().forward(blank, condition, t)


class Corrector(RadioDiffAFFPrediction3DUNet):
    def __init__(self, base_channels=32):
        super().__init__(base_channels=base_channels)
        self.input_block = ResidualBlock3D(6, base_channels)

    def forward(self, noisy_residual, condition, t, coarse):
        geometry, _ = self.aff(condition[:, 5:6], condition[:, 6:7])
        x = torch.cat((noisy_residual, condition[:, :3], geometry, coarse), dim=1)
        s0 = self.input_block(x)
        s1 = self.enc1(s0)
        x = self.enc2(s1)
        x = self.bottleneck(x + self.time_mlp(self._time_embedding(t))[:, :, None, None, None])
        return self.output(self.dec2(self.dec1(x, s1), s0))


def schedule(device, steps=200):
    # Cosine DDPM schedule: terminal alpha_bar approximately zero, not 0.13
    # as with simply shortening a 1000-step linear schedule to 200 steps.
    t = torch.arange(steps + 1, dtype=torch.float64, device=device) / steps
    a = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    a = a / a[0]
    beta = (1 - a[1:] / a[:-1]).clamp(max=0.999)
    return (1 - beta).cumprod(0).float()


def x0_from_v(xt, v, alpha):
    return alpha.sqrt() * xt - (1 - alpha).sqrt() * v


@torch.no_grad()
def sample(corrector, condition, coarse, alpha, scale, steps=30, seed=42):
    """DDIM eta=0, Gaussian initialization; target/valid mask are NOT arguments."""
    if not 1 <= steps <= len(alpha):
        raise ValueError('Sampling steps must be between 1 and diffusion timesteps')
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    x = torch.randn(coarse.shape, device=condition.device, generator=generator)
    indices = torch.linspace(len(alpha) - 1, 0, steps, device=condition.device).round().long()
    for i, t in enumerate(indices):
        a = alpha[t]
        v = corrector(x, condition, t.expand(x.shape[0]), coarse)
        clean = x0_from_v(x, v, a)
        eps = (1 - a).sqrt() * x + a.sqrt() * v
        previous = alpha[indices[i + 1]] if i + 1 < len(indices) else x.new_tensor(1.)
        x = previous.sqrt() * clean + (1 - previous).sqrt() * eps
    # Residual has signed, unbounded support: never clip it to [-1,1].
    return (coarse + scale * x).clamp(-1, 1)


def masked_mse(pred, target, valid):
    return (pred - target).square()[valid.bool()].mean()


def gradient_loss(pred, target, valid):
    terms = []
    for dim in (2, 3, 4):
        a = [slice(None)] * 5
        b = list(a)
        a[dim], b[dim] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        mask = valid[a].bool() & valid[b].bool()
        if mask.any():
            terms.append(((pred[a] - pred[b]) - (target[a] - target[b])).abs()[mask].mean())
    return torch.stack(terms).mean() if terms else pred.sum() * 0
