"""Full DDPM evaluation of the h1--h4 AFF model from pure Gaussian noise."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pl3d_manifest_dataset import PL3DManifestDataset
from diffusion.beta_schedule import get_beta_schedule
from diffusion.forward_process import DiffusionForwardProcess
from models.radiodiff_aff_full3d import (
    RadioDiffAFFFull3DUNet,
    RadioDiffAFFPrediction2p5DUNet,
    RadioDiffAFFPrediction3DUNet,
    RadioDiffAFFResidual2p5DUNet,
    RadioDiffAFFResidual3DUNet,
)
from models.radiodiff_aff_volume import RadioDiffAFFVolumeUNet


def build_aff_model_from_checkpoint(
    checkpoint: dict, base_channels: int, device: torch.device
) -> tuple[torch.nn.Module, str, str]:
    """Instantiate the recorded architecture so legacy results remain reproducible."""
    model_type = str(checkpoint.get("model_type", "pl3d_aff_volume_ddpm"))
    if model_type == RadioDiffAFFPrediction3DUNet.model_type:
        model = RadioDiffAFFPrediction3DUNet(
            base_channels=base_channels,
            aff_mode=str(checkpoint.get("aff_mode", "adaptive3d")),
            use_building_height=bool(checkpoint.get("use_building_height", True)),
        ).to(device)
        architecture = str(
            checkpoint.get("architecture", "strict_q_building_full_3d_aff_diffusion")
        )
    elif model_type == RadioDiffAFFPrediction2p5DUNet.model_type:
        model = RadioDiffAFFPrediction2p5DUNet(base_channels=base_channels).to(device)
        architecture = str(
            checkpoint.get("architecture", "strict_q_building_2p5d_aff_diffusion")
        )
    elif model_type == RadioDiffAFFResidual3DUNet.model_type:
        model = RadioDiffAFFResidual3DUNet(
            base_channels=base_channels,
            aff_mode=str(checkpoint.get("aff_mode", "adaptive3d")),
        ).to(device)
        architecture = str(
            checkpoint.get("architecture", "full_3d_aff_idw_conditioned_diffusion")
        )
    elif model_type == RadioDiffAFFResidual2p5DUNet.model_type:
        model = RadioDiffAFFResidual2p5DUNet(base_channels=base_channels).to(device)
        architecture = "2p5d_shared_2d_unet_with_height_axis_bottleneck"
    elif model_type == RadioDiffAFFFull3DUNet.model_type:
        model = RadioDiffAFFFull3DUNet(base_channels=base_channels).to(device)
        architecture = "full_3d_unet_3x3x3_xy_downsampling_only"
    elif model_type == "pl3d_aff_volume_ddpm":
        model = RadioDiffAFFVolumeUNet(base_channels=base_channels).to(device)
        architecture = "legacy_2p5d_shared_2d_unet_with_height_axis_bottleneck"
    else:
        raise ValueError(f"Unsupported AFF checkpoint model_type: {model_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_type, architecture


def compute_legacy_nmse(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Old evaluator NMSE for [B, C, H, W], reduced over the full map."""
    numerator = torch.sum((prediction - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target**2, dim=(1, 2, 3)) + eps
    return numerator / denominator


def compute_legacy_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Old evaluator RMSE for [B, C, H, W], reduced over the full map."""
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return torch.sqrt(mse)


def compute_legacy_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Old evaluator PSNR; data_range=2 matches tensors normalized to [-1, 1]."""
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


def gaussian_window(
    window_size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window_2d = torch.outer(kernel, kernel)
    window_2d = window_2d / window_2d.sum()
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Return a channel-wise 2-D SSIM map for tensors [B,C,H,W]."""
    channels = prediction.shape[1]
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(prediction**2, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target**2, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_xy
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )


def compute_legacy_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Old evaluator SSIM averaged over the complete h1--h4 volume."""
    return compute_ssim_map(prediction, target, window, data_range).mean(dim=(1, 2, 3))


def balanced_scene_indices(dataset: PL3DManifestDataset, maximum: int, seed: int) -> list[int]:
    if maximum <= 0 or maximum >= len(dataset):
        return list(range(len(dataset)))
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        groups.setdefault(int(row["scene_id"]), []).append(index)
    rng = random.Random(seed)
    scene_ids = sorted(groups)
    rng.shuffle(scene_ids)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    offset = 0
    while len(selected) < maximum:
        for scene_id in scene_ids:
            values = groups[scene_id]
            if offset < len(values):
                selected.append(values[offset])
                if len(selected) == maximum:
                    return selected
        offset += 1
    return selected


@torch.no_grad()
def sample_fullnoise(
    model: torch.nn.Module,
    diffusion: DiffusionForwardProcess,
    condition: torch.Tensor,
    measurement: torch.Tensor,
    measurement_mask: torch.Tensor,
    idw_prior: torch.Tensor,
    generator: torch.Generator,
    consistency_mode: str,
    conditioned_idw: bool,
    prediction_type: str,
    diffusion_target: str,
    residual_scale: float,
    residual_strength: float,
) -> torch.Tensor:
    """Run every reverse step from pure full-map or residual noise."""
    batch = condition.shape[0]
    shape = (batch, 1, 4, condition.shape[-2], condition.shape[-1])
    xt = torch.randn(shape, device=condition.device, generator=generator)
    known = measurement_mask.bool()
    known_noise = None
    if consistency_mode == "every_step":
        known_noise = torch.randn(shape, device=condition.device, generator=generator)
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((batch,), step, device=condition.device, dtype=torch.long)
        model_output = (
            model(xt, condition, t, idw_prior)
            if conditioned_idw
            else model(xt, condition, t)
        )
        if prediction_type == "v":
            predicted_x0 = diffusion.predict_x0_from_v(
                xt, t, model_output
            ).clamp(-1, 1)
        else:
            predicted_x0 = diffusion.predict_x0_from_noise(
                xt, t, model_output
            ).clamp(-1.0, 1.0)
        mean, _, log_variance = diffusion.q_posterior(predicted_x0, xt, t)
        if step > 0:
            reverse_noise = torch.randn(shape, device=condition.device, generator=generator)
            xt = mean + torch.exp(0.5 * log_variance) * reverse_noise
        else:
            xt = mean
        if consistency_mode == "every_step":
            if step > 0:
                previous = step - 1
                known_x0 = (
                    torch.zeros_like(measurement)
                    if diffusion_target == "normalized_idw_residual"
                    else measurement
                )
                known_xt = (
                    diffusion.sqrt_alphas_cumprod[previous] * known_x0
                    + diffusion.sqrt_one_minus_alphas_cumprod[previous] * known_noise
                )
            else:
                known_xt = (
                    torch.zeros_like(measurement)
                    if diffusion_target == "normalized_idw_residual"
                    else measurement
                )
            xt = torch.where(known, known_xt, xt)
    if diffusion_target == "normalized_idw_residual":
        prediction = (idw_prior + residual_strength * residual_scale * xt.clamp(-1, 1)).clamp(-1, 1)
    else:
        prediction = xt.clamp(-1.0, 1.0)
    return torch.where(known, measurement, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=0)
    parser.add_argument("--save-first-n", type=int, default=10)
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument(
        "--metric-protocol",
        choices=("valid_unobserved_four", "legacy_four", "valid_unobserved"),
        default="valid_unobserved_four",
        help=(
            "valid_unobserved_four is the corrected per-height four-metric protocol; "
            "legacy_four reproduces the old full-map definitions."
        ),
    )
    parser.add_argument(
        "--consistency-mode",
        choices=("final", "every_step"),
        default="every_step",
        help="final matches the old evaluator; every_step uses noisy projection each reverse step.",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    base_channels = args.base_channels or int(checkpoint.get("base_channels", 32))
    model, model_type, architecture = build_aff_model_from_checkpoint(
        checkpoint, base_channels, device
    )
    diffusion_timesteps = int(checkpoint.get("diffusion_timesteps", 200))
    beta_start = float(checkpoint.get("beta_start", 0.0005))
    beta_end = float(checkpoint.get("beta_end", 0.10))
    betas = get_beta_schedule(
        "linear", diffusion_timesteps, beta_start=beta_start, beta_end=beta_end
    )
    diffusion = DiffusionForwardProcess(betas, device=device)
    prediction_type = str(checkpoint.get("prediction_type", "epsilon"))
    diffusion_target = str(checkpoint.get("diffusion_target", "full_map"))
    conditioned_idw = model_type in {
        RadioDiffAFFResidual3DUNet.model_type,
        RadioDiffAFFResidual2p5DUNet.model_type,
    }
    residual_scale = float(checkpoint.get("residual_scale", 1.0))
    ssim_window = gaussian_window(11, 1.5, channels=4, device=device)

    condition_profile = str(checkpoint.get("condition_profile", "full"))
    evaluation_sample_rate = (
        0.0 if condition_profile == "aff_prediction" else args.sample_rate
    )
    dataset = PL3DManifestDataset(
        args.dataset_root,
        "test",
        sample_rate=evaluation_sample_rate,
        deterministic_sampling=True,
        seed=args.seed,
        condition_profile=condition_profile,
        include_idw_prior=conditioned_idw,
    )
    indices = balanced_scene_indices(dataset, args.max_samples, args.seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    print(
        f"protocol=full_random_noise device={device} samples={len(indices)} "
        f"repeats={args.repeats} reverse_steps={diffusion_timesteps} "
        f"metrics={args.metric_protocol} consistency={args.consistency_mode}",
        flush=True,
    )

    for repeat in range(args.repeats):
        for position, batch in enumerate(loader):
            target = batch["target"].to(device)
            condition = batch["condition"].to(device)
            valid = batch["valid_mask"].to(device).bool()
            measured = batch["measurement_mask"].to(device).bool()
            measurement = batch["measurement"].to(device)
            idw_prior = batch["idw_prior"].to(device)
            metric_mask = valid & ~measured
            generator = torch.Generator(device=device).manual_seed(
                args.seed + repeat * 1_000_000 + indices[position]
            )
            prediction = sample_fullnoise(
                model,
                diffusion,
                condition,
                measurement,
                measured,
                idw_prior,
                generator,
                args.consistency_mode,
                conditioned_idw,
                prediction_type,
                diffusion_target,
                residual_scale,
                args.residual_strength,
            )
            new_records: list[dict[str, float | int | str]]
            if args.metric_protocol == "legacy_four":
                # Treat h1--h4 as four image channels, exactly as the old 2-D evaluator.
                prediction_2d = prediction[:, 0].float()
                target_2d = target[:, 0].float()
                record = {
                    "id": batch["id"][0],
                    "scene_id": int(batch["scene_id"].item()),
                    "repeat": repeat,
                    "nmse": float(compute_legacy_nmse(prediction_2d, target_2d)[0].item()),
                    "rmse": float(compute_legacy_rmse(prediction_2d, target_2d)[0].item()),
                    "ssim": float(
                        compute_legacy_ssim(
                            prediction_2d, target_2d, ssim_window, data_range=2.0
                        )[0].item()
                    ),
                    "psnr": float(
                        compute_legacy_psnr(prediction_2d, target_2d, data_range=2.0)[0].item()
                    ),
                }
                progress_rmse = record["rmse"]
                new_records = [record]
            elif args.metric_protocol == "valid_unobserved_four":
                # Macro-average height slices, matching the 2-D baseline reporting unit.
                prediction_2d = prediction[:, 0].float()
                target_2d = target[:, 0].float()
                mask_2d = metric_mask[:, 0]
                # Excluded voxels are neutralized before local SSIM filtering so that
                # untrained building interiors cannot contaminate nearby valid centers.
                ssim_prediction = torch.where(mask_2d, prediction_2d, target_2d)
                ssim_maps = compute_ssim_map(
                    ssim_prediction, target_2d, ssim_window, data_range=2.0
                )
                new_records = []
                for height_index, height_m in enumerate((1, 2, 3, 4)):
                    height_mask = mask_2d[0, height_index]
                    count = height_mask.sum()
                    if int(count.item()) == 0:
                        continue
                    height_error = prediction_2d[0, height_index] - target_2d[0, height_index]
                    squared_error = height_error.square()[height_mask]
                    mse = squared_error.mean()
                    target_energy = target_2d[0, height_index].square()[height_mask].sum()
                    nmse = squared_error.sum() / target_energy.clamp_min(1e-12)
                    rmse = torch.sqrt(mse)
                    psnr = 10.0 * torch.log10(4.0 / (mse + 1e-8))
                    ssim = ssim_maps[0, height_index][height_mask].mean()
                    new_records.append(
                        {
                            "id": batch["id"][0],
                            "scene_id": int(batch["scene_id"].item()),
                            "height_m": height_m,
                            "repeat": repeat,
                            "nmse": float(nmse.item()),
                            "rmse": float(rmse.item()),
                            "ssim": float(ssim.item()),
                            "psnr": float(psnr.item()),
                            "evaluated_voxels": int(count.item()),
                        }
                    )
                if not new_records:
                    raise ValueError(f"No valid unobserved voxels for sample {batch['id'][0]}")
                progress_rmse = float(
                    np.mean([float(item["rmse"]) for item in new_records])
                )
            else:
                error = prediction - target
                count = metric_mask.sum().clamp_min(1)
                mse = (error.square() * metric_mask).sum() / count
                mae = (error.abs() * metric_mask).sum() / count
                target_energy = (target.square() * metric_mask).sum().clamp_min(1e-12)
                nmse = (error.square() * metric_mask).sum() / target_energy
                record = {
                    "id": batch["id"][0],
                    "scene_id": int(batch["scene_id"].item()),
                    "repeat": repeat,
                    "rmse_normalized": float(torch.sqrt(mse).item()),
                    "rmse_uint8": float((torch.sqrt(mse) * 127.5).item()),
                    "mae_normalized": float(mae.item()),
                    "mae_uint8": float((mae * 127.5).item()),
                    "nmse": float(nmse.item()),
                    "evaluated_voxels": int(count.item()),
                }
                progress_rmse = record["rmse_normalized"]
                new_records = [record]
            records.extend(new_records)
            if repeat == 0 and position < args.save_first_n:
                np.savez_compressed(
                    args.output_dir / f"sample_{position:04d}.npz",
                    prediction=prediction.cpu().numpy(),
                    target=target.cpu().numpy(),
                    measurement=measurement.cpu().numpy(),
                    measurement_mask=measured.cpu().numpy(),
                    idw_prior=idw_prior.cpu().numpy(),
                    valid_mask=valid.cpu().numpy(),
                )
            if (position + 1) % 10 == 0 or position + 1 == len(indices):
                print(
                    f"repeat={repeat + 1}/{args.repeats} "
                    f"sample={position + 1}/{len(indices)} "
                    f"rmse={progress_rmse:.6f}",
                    flush=True,
                )

    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.metric_protocol == "valid_unobserved_four":
        with (args.output_dir / "per_height.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if args.metric_protocol == "legacy_four":
        metric_names = ("nmse", "rmse", "ssim", "psnr")
        metric_region = "all voxels after final exact measurement overwrite (legacy protocol)"
    elif args.metric_protocol == "valid_unobserved_four":
        metric_names = ("nmse", "rmse", "ssim", "psnr")
        metric_region = "physically valid and unobserved voxels only"
    else:
        metric_names = ("rmse_normalized", "rmse_uint8", "mae_normalized", "mae_uint8", "nmse")
        metric_region = "valid and unobserved voxels only"
    training_sample_rate = float(checkpoint.get("sample_rate", 0.10))
    rate_note = (
        "checkpoint and evaluation use the same sampling rate"
        if np.isclose(training_sample_rate, evaluation_sample_rate)
        else f"checkpoint trained at {training_sample_rate:.1%}; evaluation uses {evaluation_sample_rate:.1%}"
    )
    summary: dict[str, object] = {
        "model_type": model_type,
        "architecture": architecture,
        "task": checkpoint.get("task", "random_sparse_to_dense_completion"),
        "condition_profile": condition_profile,
        "condition_names": checkpoint.get("condition_names"),
        "protocol": (
            "full_ddpm_residual_from_pure_gaussian_noise"
            if diffusion_target == "normalized_idw_residual"
            else "full_ddpm_from_pure_gaussian_noise"
        ),
        "reverse_steps": diffusion_timesteps,
        "beta_schedule": "linear",
        "beta_start": beta_start,
        "beta_end": beta_end,
        "metric_protocol": args.metric_protocol,
        "measurement_consistency": args.consistency_mode,
        "prediction_type": prediction_type,
        "diffusion_target": diffusion_target,
        "ablation_variant": checkpoint.get("ablation_variant", "legacy"),
        "residual_scale": residual_scale if diffusion_target == "normalized_idw_residual" else None,
        "residual_strength": args.residual_strength if diffusion_target == "normalized_idw_residual" else None,
        "metric_region": metric_region,
        "training_sample_rate": training_sample_rate,
        "evaluation_sample_rate": evaluation_sample_rate,
        "sampling_rate_note": rate_note,
        "test_samples": len(indices),
        "evaluated_height_slices": len(records)
        if args.metric_protocol == "valid_unobserved_four"
        else None,
        "repeats": args.repeats,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
        }
    if args.metric_protocol == "valid_unobserved_four":
        summary["aggregation"] = "macro-average over receiver-height slices"
        summary["ssim_masking"] = (
            "excluded voxels are replaced by target values before filtering; "
            "the SSIM map is averaged only at valid unobserved centers"
        )
        by_height: dict[str, object] = {}
        for height_m in (1, 2, 3, 4):
            height_rows = [row for row in records if int(row["height_m"]) == height_m]
            height_summary: dict[str, object] = {"count": len(height_rows)}
            for name in metric_names:
                values = np.asarray(
                    [float(row[name]) for row in height_rows], dtype=np.float64
                )
                height_summary[name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
                }
            by_height[str(height_m)] = height_summary
        summary["by_height_m"] = by_height
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("FULLNOISE_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
