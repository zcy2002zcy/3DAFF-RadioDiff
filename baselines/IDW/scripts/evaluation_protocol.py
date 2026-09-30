import torch
import torch.nn.functional as F


def _masked_mean(values, mask, eps=1e-8):
    mask = mask.to(dtype=values.dtype).expand_as(values)
    dims = tuple(range(1, values.ndim))
    return (values * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(eps)


def _ssim_map(pred, target, window, data_range=2.0, k1=0.01, k2=0.03):
    channels = pred.shape[1]
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x.square()
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_x * mu_y
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x2 + sigma_y2 + c2)
    )


def _region_metrics(pred, target, mask, ssim_map, data_range=2.0, eps=1e-8):
    squared_error = (pred - target).square()
    mse = _masked_mean(squared_error, mask, eps)
    target_energy = _masked_mean(target.square(), mask, eps)
    return {
        "loss": mse,
        "nmse": mse / target_energy.clamp_min(eps),
        "rmse": torch.sqrt(mse),
        "ssim": _masked_mean(ssim_map, mask, eps),
        "psnr": 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(eps)),
    }


def evaluate_regions(pred, target, building_mask, sample_mask, window, data_range=2.0):
    ones = torch.ones_like(target)
    valid = (building_mask < 0.5).to(target.dtype)
    unobserved_valid = valid * (sample_mask < 0.5).to(target.dtype)
    ssim_map = _ssim_map(pred, target, window, data_range=data_range)
    all_metrics = _region_metrics(pred, target, ones, ssim_map, data_range=data_range)
    valid_metrics = _region_metrics(pred, target, valid, ssim_map, data_range=data_range)
    unobserved_metrics = _region_metrics(
        pred, target, unobserved_valid, ssim_map, data_range=data_range
    )
    # Publication primary: full slice after the shared final observation projection.
    result = dict(all_metrics)
    result.update({f"all_{key}": value for key, value in all_metrics.items()})
    result.update({f"valid_{key}": value for key, value in valid_metrics.items()})
    result.update(
        {f"unobserved_{key}": value for key, value in unobserved_metrics.items()}
    )
    result["valid_fraction"] = valid.mean(dim=(1, 2, 3))
    result["eval_fraction"] = unobserved_valid.mean(dim=(1, 2, 3))
    return result
