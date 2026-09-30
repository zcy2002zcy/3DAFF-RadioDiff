import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.unet import UNetBaseline
from scripts.evaluation_protocol import evaluate_regions


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_input(batch):
    height, width = batch["q"].shape[-2:]
    altitude = ((batch["z"].view(-1, 1, 1, 1) - 1.0) / 3.0).clamp(0.0, 1.0)
    altitude_map = altitude.expand(-1, -1, height, width)
    return torch.cat(
        [batch["q"], batch["building_mask"], batch["s"], altitude_map],
        dim=1,
    )


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_nmse(pred, target, eps=1e-8):
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return num / den


def compute_rmse(pred, target):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return torch.sqrt(mse)


def compute_psnr(pred, target, data_range=2.0, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim(pred, target, window, data_range=2.0, K1=0.01, K2=0.03):
    B, C, H, W = pred.shape
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=C)
    mu_y = F.conv2d(target, window, padding=padding, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=C) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Config:", cfg_path)
    print("Using device:", device)

    model_cfg = cfg.get("model", {})
    model = UNetBaseline(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    ).to(device)

    ckpt_dir = resolve_path(cfg["checkpoint"]["save_dir"])
    ckpt_path = ckpt_dir / ("best.pt" if args.checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Checkpoint:", ckpt_path)

    loader = build_dataloader_from_config(cfg["data"]["test_dir"], cfg, shuffle=False, drop_last=False, split="test")
    save_dir = resolve_path(cfg["eval"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_first_n = int(cfg.get("eval", {}).get("save_first_n", 20))
    use_amp = bool(cfg.get("train", {}).get("amp", True)) and device.type == "cuda"

    rows = []
    metric_names = [
        "loss", "nmse", "rmse", "ssim", "psnr",
        "all_loss", "all_nmse", "all_rmse", "all_ssim", "all_psnr",
        "valid_loss", "valid_nmse", "valid_rmse", "valid_ssim", "valid_psnr",
        "valid_fraction", "eval_fraction",
    ]
    totals = {name: 0.0 for name in metric_names}
    total_count = 0
    window = None

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = move_batch(batch, device)
            x = build_input(batch)
            target = batch["psi"]
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)

            pred_metric = pred.float().clamp(-1.0, 1.0)
            pred_metric = torch.where(batch["s_mask"] > 0.5, batch["s"], pred_metric)
            target_metric = target.float()
            if window is None:
                window = gaussian_window(
                    window_size=11,
                    sigma=1.5,
                    channels=target_metric.shape[1],
                    device=device,
                    dtype=target_metric.dtype,
                )

            metrics = evaluate_regions(
                pred_metric, target_metric, batch["building_mask"], batch["s_mask"], window, data_range=2.0
            )

            B = target.shape[0]
            for name in metric_names:
                totals[name] += float(metrics[name].sum().detach().cpu())
            total_count += B

            pred_np = pred_metric.detach().cpu().numpy()
            target_np = target_metric.detach().cpu().numpy()
            q_np = batch["q"].detach().float().cpu().numpy()
            building_mask_np = batch["building_mask"].detach().float().cpu().numpy()
            s = batch.get("s")
            s_np = s.detach().float().cpu().numpy() if torch.is_tensor(s) else None
            ids = batch.get("id", [str(idx)])
            ids_list = list(ids) if isinstance(ids, (list, tuple)) else [str(ids)] * B

            metric_np = {name: metrics[name].detach().cpu().numpy() for name in metric_names}

            for b in range(B):
                row = {
                    "index": len(rows),
                    "id": ids_list[b],
                    **{name: float(metric_np[name][b]) for name in metric_names},
                }
                rows.append(row)
                if len(rows) <= save_first_n:
                    out = {
                        "prediction": pred_np[b, 0],
                        "target": target_np[b, 0],
                        "q": q_np[b, 0],
                        "building_mask": building_mask_np[b, 0],
                    }
                    if s_np is not None:
                        out["sampled"] = s_np[b, 0]
                    np.savez_compressed(save_dir / f"sample_{len(rows)-1:04d}.npz", **out)

    summary = {name: totals[name] / max(total_count, 1) for name in metric_names}
    summary["count"] = total_count

    with (save_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (save_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n========== Eval Summary: Full Slice After Measurement Projection ==========")
    print(f"Eval fraction : {summary['eval_fraction']:.6f}")
    print(f"Eval Loss : {summary['loss']:.6f}")
    print(f"Eval NMSE : {summary['nmse']:.6f}")
    print(f"Eval RMSE : {summary['rmse']:.6f}")
    print(f"Eval SSIM : {summary['ssim']:.6f}")
    print(f"Eval PSNR : {summary['psnr']:.6f}")
    print("---------- Full-Slice Compatibility Fields ----------")
    print(f"All NMSE : {summary['all_nmse']:.6f}")
    print(f"All RMSE : {summary['all_rmse']:.6f}")
    print(f"All SSIM : {summary['all_ssim']:.6f}")
    print(f"All PSNR : {summary['all_psnr']:.6f}")
    print(f"Count     : {summary['count']}")
    print("==================================")
    print("Saved:", save_dir)


if __name__ == "__main__":
    main()
