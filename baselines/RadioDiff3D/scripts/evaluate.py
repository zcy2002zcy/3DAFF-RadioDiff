import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import RadioDiff3DVolumeDataset
from diffusion import GaussianDiffusion
from utils.common import build_condition, build_model, load_config, move_batch, project_path, seed_everything
from utils.metrics import metrics_per_slice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--max-volumes", type=int, default=-1)
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument(
        "--consistency-mode",
        choices=("none", "projection_final", "projection_each_step"),
        default="",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    max_volumes = eval_cfg["max_volumes"] if args.max_volumes < 0 else args.max_volumes
    dataset = RadioDiff3DVolumeDataset(
        str(project_path(data_cfg["manifest_dir"]) / "test.json"),
        building_dir=data_cfg["building_dir"], image_size=data_cfg["image_size"],
        max_building_height=data_cfg["max_building_height"],
        use_sampling=data_cfg["use_sampling"], sample_rate=data_cfg["sample_rate"], deterministic_sampling=True,
        seed=seed, max_volumes=max_volumes
    )
    loader = DataLoader(
        dataset, batch_size=eval_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=data_cfg["pin_memory"]
    )

    checkpoint_path = project_path(cfg["train"]["output_dir"]) / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    weights = checkpoint.get("model_ema", checkpoint.get("ema", checkpoint["model"]))
    model.load_state_dict(weights)
    model.eval()
    dcfg = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=dcfg["timesteps"], schedule=dcfg["beta_schedule"],
        beta_start=dcfg["beta_start"], beta_end=dcfg["beta_end"],
        cosine_s=dcfg.get("cosine_s", 0.008),
    ).to(device)
    consistency_mode = (
        args.consistency_mode
        if args.consistency_mode
        else str(eval_cfg.get("measurement_consistency", "none"))
    )
    enforce_consistency = (
        bool(data_cfg["use_sampling"])
        and consistency_mode in {"projection_final", "projection_each_step"}
        and not args.no_consistency
    )
    print(f"Checkpoint: {checkpoint_path}")
    print("Protocol: x_T ~ N(0,I), deterministic DDIM reverse sampling")
    print(f"Training T={diffusion.timesteps}, DDIM steps={cfg['diffusion']['ddim_steps']}")
    print(f"Volumes={len(dataset)}, slices={len(dataset) * len(data_cfg['altitudes'])}")
    print(f"Sparse measurement consistency={consistency_mode if enforce_consistency else 'none'}")

    output_dir = project_path(args.output_dir or eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {name: 0.0 for name in ("mse", "nmse", "rmse", "ssim", "psnr")}
    totals_before = {name: 0.0 for name in totals}
    total_slices = 0
    rows = []
    with torch.inference_mode():
        for volume_index, batch in enumerate(tqdm(loader, desc="DDIM eval")):
            batch = move_batch(batch, device)
            target = batch["psi"]
            generator = torch.Generator(device=device).manual_seed(seed + volume_index)
            initial_noise = torch.randn(target.shape, device=device, generator=generator)
            condition = build_condition(batch, cfg)
            if torch.any(batch["s"][batch["s_mask"] == 0] != 0):
                raise RuntimeError("Sparse target leakage detected before evaluation")
            prediction = diffusion.ddim_sample(
                model=model, condition=condition, shape=target.shape,
                steps=cfg["diffusion"]["ddim_steps"], eta=cfg["diffusion"]["ddim_eta"],
                clip_denoised=cfg["diffusion"]["clip_denoised"], initial_noise=initial_noise,
                observed=batch["s"] if enforce_consistency else None,
                observed_mask=batch["s_mask"] if enforce_consistency else None,
                project_observations_each_step=(
                    enforce_consistency and consistency_mode == "projection_each_step"
                ),
            )
            before_values = metrics_per_slice(prediction, target)
            if enforce_consistency and consistency_mode == "projection_final":
                prediction = torch.where(batch["s_mask"] > 0.5, batch["s"], prediction)
            values = metrics_per_slice(prediction, target)
            batch_slices = values["mse"].numel()
            total_slices += batch_slices
            for name in totals:
                totals[name] += float(values[name].sum())
                totals_before[name] += float(before_values[name].sum())
            batch_ids = list(batch["id"])
            depth_count = target.shape[2]
            for flat_index in range(batch_slices):
                batch_index = flat_index // depth_count
                depth_index = flat_index % depth_count
                row = {
                    "id": batch_ids[batch_index],
                    "altitude": float(batch["z"][batch_index, depth_index].cpu()),
                }
                row.update({name: float(values[name][flat_index]) for name in totals})
                rows.append(row)
            if volume_index < eval_cfg["save_first_n"]:
                np.savez_compressed(
                    output_dir / f"volume_{volume_index:04d}.npz",
                    id=np.asarray(batch_ids), prediction=prediction.cpu().numpy(),
                    ground_truth=target.cpu().numpy(), q=batch["q"].cpu().numpy(),
                    building_mask=batch["building_mask"].cpu().numpy(),
                    building_height=batch["building_height"].cpu().numpy(),
                    sparse=batch["s"].cpu().numpy(),
                    sparse_mask=batch["s_mask"].cpu().numpy(),
                    initial_noise=initial_noise.cpu().numpy()
                )

    summary = {name: total / max(total_slices, 1) for name, total in totals.items()}
    summary.update({
        f"{name}_before_consistency": total / max(total_slices, 1)
        for name, total in totals_before.items()
    })
    summary.update(
        {
            "count_slices": total_slices,
            "count_volumes": len(dataset),
            "sample_rate": data_cfg["sample_rate"],
            "ddim_steps": cfg["diffusion"]["ddim_steps"],
            "measurement_consistency": enforce_consistency,
            "measurement_consistency_mode": consistency_mode if enforce_consistency else "none",
        }
    )
    with (output_dir / "metrics_per_slice.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 58)
    print("RadioDiff-3D reimplementation / PL / q + building + random 10% samples")
    print(f"NMSE : {summary['nmse']:.6f}")
    print(f"RMSE : {summary['rmse']:.6f}")
    print(f"SSIM : {summary['ssim']:.6f}")
    print(f"PSNR: {summary['psnr']:.6f} dB")
    if enforce_consistency and consistency_mode == "projection_final":
        print("Before final measurement projection:")
        print(f"  NMSE : {summary['nmse_before_consistency']:.6f}")
        print(f"  RMSE : {summary['rmse_before_consistency']:.6f}")
        print(f"  SSIM : {summary['ssim_before_consistency']:.6f}")
        print(f"  PSNR : {summary['psnr_before_consistency']:.6f} dB")
    elif enforce_consistency and consistency_mode == "projection_each_step":
        print("Measurement consistency: applied throughout all reverse steps")
    print(f"Count: {total_slices} slices / {len(dataset)} volumes")
    print(f"Saved: {output_dir}")
    print("=" * 58)


if __name__ == "__main__":
    main()
