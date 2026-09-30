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
from models.pix2pix import Pix2PixGenerator
from scripts.metrics import compute_nmse, compute_psnr, compute_rmse, compute_ssim, gaussian_window
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best")
    args = parser.parse_args()
    cfg = load_yaml(resolve_path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    model = Pix2PixGenerator(mcfg["in_channels"], mcfg["out_channels"], mcfg["base_channels"]).to(device)
    ckpt_path = resolve_path(cfg["checkpoint"]["save_dir"]) / ("best.pt" if args.checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()
    print("Checkpoint:", ckpt_path)

    loader = build_dataloader_from_config(cfg["data"]["test_dir"], cfg, shuffle=False, drop_last=False, split="test")
    save_dir = resolve_path(cfg["eval"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_first_n = int(cfg["eval"].get("save_first_n", 20))
    metric_names = ["loss", "nmse", "rmse", "ssim", "psnr", "all_loss", "all_nmse", "all_rmse", "all_ssim", "all_psnr", "valid_loss", "valid_nmse", "valid_rmse", "valid_ssim", "valid_psnr", "valid_fraction", "eval_fraction"]
    totals = {name: 0.0 for name in metric_names}
    rows = []
    count = 0
    window = None
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = move_batch(batch, device)
            x = build_input(batch)
            y = batch["psi"].float()
            pred = model(x).float().clamp(-1.0, 1.0)
            pred = torch.where(batch["s_mask"] > 0.5, batch["s"], pred)
            if window is None:
                window = gaussian_window(channels=y.shape[1], device=device, dtype=y.dtype)
            metrics = evaluate_regions(pred, y, batch["building_mask"], batch["s_mask"], window)
            bs = y.shape[0]
            count += bs
            for k in metric_names:
                totals[k] += metrics[k].sum().item()
            ids = batch.get("id", [str(idx)])
            ids = list(ids) if isinstance(ids, (list, tuple)) else [str(ids)] * bs
            for b in range(bs):
                rows.append({"index": len(rows), "id": ids[b], **{k: metrics[k][b].item() for k in metric_names}})
                if len(rows) <= save_first_n:
                    np.savez_compressed(
                        save_dir / f"sample_{len(rows)-1:04d}.npz",
                        prediction=pred[b, 0].cpu().numpy(),
                        target=y[b, 0].cpu().numpy(),
                        q=batch["q"][b, 0].cpu().numpy(),
                        building_mask=batch["building_mask"][b, 0].cpu().numpy(),
                        sampled=batch["s"][b, 0].cpu().numpy(),
                        sample_mask=batch["s_mask"][b, 0].cpu().numpy(),
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
