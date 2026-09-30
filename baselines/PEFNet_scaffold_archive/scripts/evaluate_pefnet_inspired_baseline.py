"""Evaluate the PEFNet-inspired adapted baseline on the common test protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pefnet_inspired_dataset import PEFNetInspiredDataset
from models.pefnet_inspired_baseline import build_pefnet_inspired
from scripts.evaluate import metrics, normalized_metrics
from utils.config import load_config


METHOD_LABEL = "PEFNet-inspired VIE-regularized U-Net (adapted)"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pefnet_inspired_adapted.yaml")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config_path = PROJECT_ROOT / args.config if not args.config.is_absolute() else args.config
    config, _ = load_config(config_path)
    experiment = config.get("experiment", {})
    if not isinstance(experiment, dict):
        raise TypeError("Config section 'experiment' must be a mapping")
    experiment_name = str(
        experiment.get("name", "pefnet_inspired_adapted_stable_v2_random0")
    )
    name = f"{experiment_name}_seed{args.seed}"
    if args.smoke_test:
        name += "_smoke"
    output = PROJECT_ROOT / "outputs" / name
    completion_path = output / "run_complete.json"
    if not completion_path.is_file():
        failed_path = output / "run_failed.json"
        detail = f"; failure record: {failed_path}" if failed_path.is_file() else ""
        raise RuntimeError(
            f"Refusing to evaluate an incomplete PEFNet-inspired run: "
            f"{completion_path} is missing{detail}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if int(completion.get("seed", -1)) != args.seed:
        raise ValueError(f"Completion seed mismatch in {completion_path}")
    if int(completion.get("data_seed", -1)) != args.data_seed:
        raise ValueError(f"Completion data_seed mismatch in {completion_path}")
    if completion.get("stop_reason") not in {"early_stopping", "max_epochs"}:
        raise ValueError(f"Invalid completion stop_reason in {completion_path}")
    resolved_config_path = output / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(resolved_config_path)
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    resolved_digest = hashlib.sha256(
        json.dumps(
            resolved_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if completion.get("config_sha256") != resolved_digest:
        raise ValueError(
            f"Completion/resolved-config hash mismatch in {completion_path}"
        )
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pefnet_inspired(
        in_channels=int(model_config.get("in_channels", 5)),
        base_channels=int(model_config.get("base_channels", config.get("model", {}).get("base_channels", 128))),
        max_norm_groups=int(model_config.get("max_norm_groups", 32)),
        force_refinement_fp32=bool(
            model_config.get("force_refinement_fp32", True)
        ),
    )
    if model_config.get("normalization") != "group_norm":
        raise ValueError(
            "Stable-v2 evaluation requires a GroupNorm checkpoint; refusing legacy BatchNorm state"
        )
    if int(checkpoint.get("epoch", -1)) != int(completion.get("best_epoch", -2)):
        raise ValueError("best.pt epoch does not match run_complete best_epoch")
    checkpoint_metadata = checkpoint.get("metadata", {})
    if checkpoint_metadata.get("config_sha256") != resolved_digest:
        raise ValueError("Checkpoint/resolved-config hash mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    physics = config.get("physics", {})
    data = config.get("data", {})
    dataset = PEFNetInspiredDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=2 if args.smoke_test else int(data.get("test_maps", 128)),
        selection_strategy="sequential" if args.smoke_test else str(data.get("selection_strategy", "balanced_scene")),
        seed=args.data_seed,
        training=False,
        dimensionless_wavenumber=float(physics.get("dimensionless_wavenumber", 0.35)),
        grid_spacing=float(physics.get("grid_spacing", 1.0)),
        epsilon_air=float(physics.get("epsilon_air", 1.0)),
        epsilon_building=float(physics.get("epsilon_building", 4.0)),
        conductivity=float(physics.get("conductivity_s_per_m", 0.0)),
    )
    rows: list[dict[str, object]] = []
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            prediction = (
                model(batch["input"].to(device, non_blocking=True))["prediction"]
                .float().cpu().numpy()[0, 0] * 255.0
            )
            target = batch["target_raw"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
            observed = batch["observation_mask"].numpy()[0, 0].astype(bool)
            support = valid & ~observed
            if not support.any():
                raise ValueError(f"Test map {index} has no valid receiver support")
            encoded = normalized_metrics(prediction[support], target[support])
            raw = metrics(prediction[support], target[support], 255.0)
            row: dict[str, object] = {
                "map_index": index,
                "scene_id": int(batch["scene_id"].item()),
                "valid_points": int(valid.sum()),
                "observed_points": int(observed.sum()),
                **encoded,
                **{f"raw_{key}": value for key, value in raw.items()},
            }
            rows.append(row)
            print(
                f"[{index + 1}/{len(dataset)}] rmse={encoded['rmse']:.4f} "
                f"nmse={encoded['nmse']:.6f}"
            )
    if not rows:
        raise ValueError("Test dataset is empty")
    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {}
    for metric_name in (
        "rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr"
    ):
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        summary[metric_name] = {"mean": float(values.mean()), "std": float(values.std())}
    summary.update(
        {
            "maps": len(rows),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "sample_rate": 0.0,
            "support": "valid_rx_only",
            "primary_domain": "encoded normalized x / 127.5 - 1.0",
            "primary_metric_domain": "normalized_minus1_to_1",
            "normalization": "prediction and target mapped from PNG code x to x / 127.5 - 1.0",
            "raw_domain": "UrbanRadio3D PNG code 0..255; not asserted to be dB",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "config_sha256": resolved_digest,
            "stability_revision": completion.get("stability_revision"),
            "runtime_precision": completion.get("runtime_precision"),
            "normalization_layer": "group_norm",
            "force_refinement_fp32": bool(
                completion.get("force_refinement_fp32", True)
            ),
            "method_label": METHOD_LABEL,
            "implementation_status": "independent_inspired_adaptation",
            "faithful_reproduction": False,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    (result / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
