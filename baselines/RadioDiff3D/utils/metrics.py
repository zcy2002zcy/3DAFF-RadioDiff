import torch
from torch.nn import functional as F


def _window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.arange(11, device=device, dtype=dtype) - 5
    kernel = torch.exp(-(x**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel).view(1, 1, 11, 11)
    return window.expand(channels, 1, 11, 11).contiguous()


def metrics_per_slice(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Use the exact per-slice metric definitions of 3DAFF-RadioDiff."""
    pred = prediction.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
    truth = target.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
    dims = (1, 2, 3)
    mse = ((pred - truth) ** 2).mean(dim=dims)
    rmse = mse.sqrt()
    nmse = ((pred - truth) ** 2).sum(dim=dims) / truth.square().sum(dim=dims).clamp_min(1e-12)
    psnr = 10.0 * torch.log10(torch.tensor(4.0, device=pred.device) / mse.clamp_min(1e-12))

    window = _window(pred.shape[1], pred.device, pred.dtype)
    mu_x = F.conv2d(pred, window, padding=5, groups=pred.shape[1])
    mu_y = F.conv2d(truth, window, padding=5, groups=truth.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(pred.square(), window, padding=5, groups=pred.shape[1]) - mu_x2
    sigma_y2 = F.conv2d(truth.square(), window, padding=5, groups=truth.shape[1]) - mu_y2
    sigma_xy = F.conv2d(pred * truth, window, padding=5, groups=pred.shape[1]) - mu_xy
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    c3 = c2 / 2.0
    luminance = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1 + 1e-12)
    sigma_x = torch.sqrt(torch.clamp(sigma_x2, min=0.0) + 1e-12)
    sigma_y = torch.sqrt(torch.clamp(sigma_y2, min=0.0) + 1e-12)
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x2 + sigma_y2 + c2 + 1e-12)
    structure = (sigma_xy + c3) / (sigma_x * sigma_y + c3 + 1e-12)
    ssim_map = luminance * contrast * structure
    ssim = ssim_map.mean(dim=dims)
    return {"mse": mse, "nmse": nmse, "rmse": rmse, "ssim": ssim, "psnr": psnr}
