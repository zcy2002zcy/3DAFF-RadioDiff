import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from scripts.metrics import compute_nmse, compute_psnr, compute_rmse, compute_ssim, gaussian_window
from scripts.evaluation_protocol import evaluate_regions

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def idw_numpy(sampled, sample_mask, k=8, power=2.0, eps=1e-8):
    mask = sample_mask.astype(bool)
    h, w = sampled.shape
    if mask.sum() == 0:
        return np.zeros_like(sampled, dtype=np.float32)
    points = np.column_stack(np.nonzero(mask)).astype(np.float32)
    values = sampled[mask].astype(np.float32)
    grid = np.indices((h, w)).reshape(2, -1).T.astype(np.float32)
    if cKDTree is not None:
        tree = cKDTree(points)
        dist, idx = tree.query(grid, k=min(k, len(points)), workers=-1)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        weights = 1.0 / (np.power(dist, power) + eps)
        pred = (weights * values[idx]).sum(axis=1) / weights.sum(axis=1)
        exact = dist[:, 0] < eps
        if exact.any():
            pred[exact] = values[idx[exact, 0]]
        return pred.reshape(h, w).astype(np.float32)

    pred = np.empty((h * w,), dtype=np.float32)
    chunk = 2048
    kk = min(k, len(points))
    for start in range(0, len(grid), chunk):
        g = grid[start:start + chunk]
        d = np.sqrt(((g[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dsel = np.take_along_axis(d, idx, axis=1)
        wgt = 1.0 / (np.power(dsel, power) + eps)
        pred[start:start + len(g)] = (wgt * values[idx]).sum(axis=1) / wgt.sum(axis=1)
    return pred.reshape(h, w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    loader = build_dataloader_from_config(cfg["data"]["test_dir"], cfg, shuffle=False, drop_last=False, split="test")
    save_dir = resolve_path(cfg["eval"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_first_n = int(cfg["eval"].get("save_first_n", 20))
    k = int(cfg["model"].get("k_neighbors", 8))
    power = float(cfg["model"].get("power", 2.0))
    eps = float(cfg["model"].get("eps", 1e-8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = ["loss", "nmse", "rmse", "ssim", "psnr", "all_loss", "all_nmse", "all_rmse", "all_ssim", "all_psnr", "valid_loss", "valid_nmse", "valid_rmse", "valid_ssim", "valid_psnr", "valid_fraction", "eval_fraction"]
    totals = {name: 0.0 for name in metric_names}
    rows = []
    count = 0
    window = None
    for idx, batch in enumerate(tqdm(loader, desc="Eval IDW")):
        target = batch["psi"].float().to(device)
        preds = []
        for b in range(target.shape[0]):
            sampled = batch["s"][b, 0].numpy()
            sample_mask = batch["s_mask"][b, 0].numpy()
            preds.append(idw_numpy(sampled, sample_mask, k=k, power=power, eps=eps))
        pred = torch.from_numpy(np.stack(preds)[:, None]).float().to(device)
        sample_values = batch["s"].float().to(device)
        sample_mask = batch["s_mask"].float().to(device)
        pred = pred.clamp(-1.0, 1.0)
        pred = torch.where(sample_mask > 0.5, sample_values, pred)
        if window is None:
            window = gaussian_window(channels=target.shape[1], device=device, dtype=target.dtype)
        building_mask = batch["building_mask"].float().to(device)
        metrics = evaluate_regions(pred, target, building_mask, sample_mask, window)
        bs = target.shape[0]
        count += bs
        for name in metric_names:
            totals[name] += metrics[name].sum().item()
        ids = batch.get("id", [str(idx)])
        ids = list(ids) if isinstance(ids, (list, tuple)) else [str(ids)] * bs
        for b in range(bs):
            rows.append({"index": len(rows), "id": ids[b], **{name: metrics[name][b].item() for name in metric_names}})
            if len(rows) <= save_first_n:
                np.savez_compressed(
                    save_dir / f"sample_{len(rows)-1:04d}.npz",
                    prediction=pred[b, 0].cpu().numpy(),
                    target=target[b, 0].cpu().numpy(),
                    q=batch["q"][b, 0].numpy(),
                    sampled=batch["s"][b, 0].numpy(),
                    sample_mask=batch["s_mask"][b, 0].numpy(),
                )
    summary = {k: v / max(count, 1) for k, v in totals.items()}
    summary["count"] = count
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
