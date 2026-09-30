import torch
import torch.nn.functional as F


def compute_nmse(pred, target, eps=1e-8):
    return torch.sum((pred - target) ** 2, dim=(1, 2, 3)) / (torch.sum(target ** 2, dim=(1, 2, 3)) + eps)


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2, 3)))


def compute_psnr(pred, target, data_range=2.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = torch.outer(g, g)
    w = w / w.sum()
    return w.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim(pred, target, window, data_range=2.0, K1=0.01, K2=0.03):
    c = pred.shape[1]
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=c)
    mu_y = F.conv2d(target, window, padding=padding, groups=c)
    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=c) - mu_x.pow(2)
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=c) - mu_y.pow(2)
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=c) - mu_x * mu_y
    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))
