import torch
import torch.nn.functional as F


def compute_nmse(pred, target, eps=1e-8):
    dims = (1, 2, 3)
    return torch.sum((pred - target) ** 2, dim=dims) / (torch.sum(target ** 2, dim=dims) + eps)


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2, 3)))


def compute_psnr(pred, target, data_range=2.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim(pred, target, window, data_range=2.0, k1=0.01, k2=0.03):
    channels = pred.shape[1]
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(pred.square(), window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean(dim=(1, 2, 3))
