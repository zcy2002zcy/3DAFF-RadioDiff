import math
from typing import Optional

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = values.gather(0, t)
    return result.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 200, schedule: str = "linear",
                 beta_start: float = 0.0005, beta_end: float = 0.10,
                 cosine_s: float = 0.008) -> None:
        self.timesteps = int(timesteps)
        if schedule == "linear":
            self.betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(self.timesteps, cosine_s)
        else:
            raise ValueError(f"Unsupported beta schedule: {schedule}")
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t, x0.shape)
        xt = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        return xt, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        shape: torch.Size,
        steps: int = 200,
        eta: float = 0.0,
        clip_denoised: bool = True,
        initial_noise: Optional[torch.Tensor] = None,
        observed: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        project_observations_each_step: bool = False,
    ) -> torch.Tensor:
        device = condition.device
        x = torch.randn(shape, device=device) if initial_noise is None else initial_noise.clone()
        if project_observations_each_step:
            if observed is None or observed_mask is None:
                raise ValueError("Stepwise projection requires observed values and a mask")
            observation_noise = x.clone()
            terminal_alpha = self.alpha_bar[self.timesteps - 1]
            noisy_observation = (
                terminal_alpha.sqrt() * observed
                + (1.0 - terminal_alpha).sqrt() * observation_noise
            )
            x = torch.where(observed_mask > 0.5, noisy_observation, x)
        times = torch.linspace(0, self.timesteps - 1, steps, device=device).long().unique()
        for position in reversed(range(len(times))):
            time = int(times[position].item())
            previous = int(times[position - 1].item()) if position > 0 else -1
            t = torch.full((shape[0],), time, device=device, dtype=torch.long)
            predicted_noise = model(x, condition, t)
            alpha = self.alpha_bar[time]
            alpha_previous = self.alpha_bar[previous] if previous >= 0 else torch.tensor(1.0, device=device)
            predicted_x0 = (x - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if clip_denoised:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha))
                * (1.0 - alpha / alpha_previous)
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_previous - sigma**2, min=0.0)) * predicted_noise
            noise = torch.randn_like(x) if previous >= 0 and eta > 0 else 0.0
            x = alpha_previous.sqrt() * predicted_x0 + direction + sigma * noise
            if project_observations_each_step:
                if previous >= 0:
                    noisy_observation = (
                        alpha_previous.sqrt() * observed
                        + (1.0 - alpha_previous).sqrt() * observation_noise
                    )
                else:
                    noisy_observation = observed
                x = torch.where(observed_mask > 0.5, noisy_observation, x)

        return x
