"""Evaluate best.pt on scene-disjoint test maps and reconstruct full grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coordinate_query_dataset import CoordinateQueryDataset
from models.continuous_radio_field import ContinuousRadioField
from utils.config import load_config
from utils.visualization import save_reconstruction_figure


METRIC_FIELDS = (
    "map_index", "scene_id", "tx_x", "tx_y", "rx_height", "valid_points", "observed_points", "evaluated_points",
    "rmse", "nmse", "mae", "psnr",
    "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr",
)

METRIC_NAMES = ("rmse", "nmse", "mae", "psnr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--no-save-arrays", action="store_true")
    parser.add_argument("--no-save-figures", action="store_true")
    return parser.parse_args()


def resolve_path(value: object, config_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def experiment_name(config: dict[str, Any]) -> str:
    name = str(config["experiment"]["name"])
    tag = str(config["training"].get("run_tag", "")).strip()
    return f"{name}_{tag}" if tag else name


def build_model(config: dict[str, Any]) -> ContinuousRadioField:
    experiment = config["experiment"]
    model_config = config["model"]
    geometry_dim = int(bool(experiment["use_los"])) + int(bool(experiment["use_depth"]))
    sparse = config.get("sparse_calibration", {})
    calibration_dim = (
        2 * len(sparse.get("context_radii", (4, 16, 64))) + 1
        if float(sparse.get("sample_rate", 0.0)) > 0.0
        else 0
    )
    return ContinuousRadioField(
        num_frequencies=int(model_config["num_frequencies"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        geometry_input_dim=geometry_dim,
        geometry_hidden_dim=int(model_config["geometry_hidden_dim"]),
        geometry_output_dim=int(model_config["geometry_output_dim"]),
        use_layer_norm=bool(model_config["use_layer_norm"]),
        xy_scale=float(model_config["xy_scale"]),
        z_scale=float(model_config["z_scale"]),
        distance_scale=float(model_config["distance_scale"]),
        calibration_input_dim=calibration_dim,
        calibration_hidden_dim=int(model_config.get("calibration_hidden_dim", 32)),
    )


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float,
) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(error ** 2))
    target_power = max(float(np.mean(target.astype(np.float64) ** 2)), 1e-12)
    return {
        "rmse": math.sqrt(mse),
        "nmse": mse / target_power,
        "mae": float(np.mean(np.abs(error))),
        "psnr": math.inf if mse == 0 else 20.0 * math.log10(data_range / math.sqrt(mse)),
    }


def normalized_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Match the legacy project's [-1, 1] metric value domain."""
    prediction_normalized = prediction.astype(np.float64) / 127.5 - 1.0
    target_normalized = target.astype(np.float64) / 127.5 - 1.0
    return metrics(prediction_normalized, target_normalized, data_range=2.0)


@torch.inference_mode()
def predict_map(
    model: ContinuousRadioField,
    sample: dict[str, torch.Tensor],
    device: torch.device,
    query_batch_size: int,
    target_scale: float,
) -> np.ndarray:
    coordinates = sample["coordinate_features"]
    geometry = sample.get("geometry_features")
    sparse_context = sample.get("sparse_context_features")
    predictions = []
    for start in range(0, len(coordinates), query_batch_size):
        end = min(start + query_batch_size, len(coordinates))
        block_geometry = (
            geometry[start:end].to(device) if geometry is not None else None
        )
        block_context = (
            sparse_context[start:end].to(device) if sparse_context is not None else None
        )
        block = model(coordinates[start:end].to(device), block_geometry, block_context)
        predictions.append((block.squeeze(-1) * target_scale).cpu())
    return torch.cat(predictions).numpy()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    name = experiment_name(config)
    output_root = resolve_path(config["output_root"], config_path) / name
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output_root / "checkpoints" / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    evaluation = config["evaluation"]
    max_maps = int(evaluation["test_max_maps"] if args.max_maps is None else args.max_maps)
    data_root = resolve_path(config["data"]["output_root"], config_path)
    cache_root = resolve_path(config["geometry"]["cache_root"], config_path)
    experiment = config["experiment"]
    sparse = config.get("sparse_calibration", {})
    dataset = CoordinateQueryDataset(
        manifest_path=data_root / "splits" / "test.csv",
        points_per_map=1,
        query_mode="full_grid",
        seed=int(config["project"]["seed"]),
        selection_seed=int(config["training"].get("data_seed", config["project"]["seed"])),
        pathloss_channel=int(config["data"]["pathloss_channel"]),
        max_records=max_maps,
        selection_strategy=str(evaluation["selection_strategy"]),
        geometry_cache_root=cache_root,
        use_los=bool(experiment["use_los"]),
        use_depth=bool(experiment["use_depth"]),
        depth_feature=str(experiment["depth_feature"]),
        exclude_building_occupied=bool(config["dataset"]["exclude_building_occupied"]),
        exclude_target_values=config["dataset"]["exclude_target_values"],
        target_exclusion_tolerance=float(config["dataset"]["target_exclusion_tolerance"]),
        sparse_sample_rate=float(sparse.get("sample_rate", 0.0)),
        sparse_context_radii=sparse.get("context_radii", (4, 16, 64)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    result_root = output_root / "evaluation" / "test"
    sample_root = result_root / "samples"
    result_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    save_arrays = bool(evaluation["save_arrays"]) and not args.no_save_arrays
    save_figures = bool(evaluation["save_figures"]) and not args.no_save_figures
    for index in range(len(dataset)):
        sample = dataset[index]
        record = dataset.records[index]
        target = sample["target"].squeeze(-1).numpy()
        valid = sample["valid_mask"].squeeze(-1).numpy().astype(bool)
        observed = sample.get("observation_mask")
        observed = (
            observed.squeeze(-1).numpy().astype(bool)
            if observed is not None
            else np.zeros_like(valid)
        )
        evaluated = valid & ~observed
        prediction = predict_map(
            model,
            sample,
            device,
            int(evaluation["query_batch_size"]),
            float(config["training"]["target_scale"]),
        )
        height = int(round(math.sqrt(len(target))))
        width = len(target) // height
        if height * width != len(target):
            raise ValueError(f"Cannot infer raster shape from {len(target)} queries")
        # The primary paper-facing metrics use the legacy [-1, 1] value domain,
        # while retaining the stricter valid-Rx evaluation support. Raw PNG-scale
        # metrics are exported alongside them for auditability.
        score = normalized_metrics(prediction[evaluated], target[evaluated])
        raw_score = metrics(prediction[evaluated], target[evaluated], data_range=255.0)
        row = {
            "map_index": index,
            "scene_id": record.scene_id,
            "tx_x": record.tx_x,
            "tx_y": record.tx_y,
            "rx_height": record.rx_height,
            "valid_points": int(valid.sum()),
            "observed_points": int(observed.sum()),
            "evaluated_points": int(evaluated.sum()),
            **score,
            **{f"raw_{key}": value for key, value in raw_score.items()},
        }
        rows.append(row)
        key = (
            f"map_{index:04d}_scene_{record.scene_id:04d}_"
            f"tx_{record.tx_x:g}_{record.tx_y:g}_z{record.rx_height:g}"
        )
        directory = sample_root / key
        target_map = target.reshape(height, width)
        prediction_map = prediction.reshape(height, width)
        valid_map = valid.reshape(height, width)
        if save_arrays:
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "gt.npy", target_map)
            np.save(directory / "prediction.npy", prediction_map)
            np.save(
                directory / "error.npy",
                np.where(valid_map, np.abs(prediction_map - target_map), np.nan),
            )
            np.save(directory / "valid_mask.npy", valid_map)
            np.save(directory / "observation_mask.npy", observed.reshape(height, width))
        if save_figures:
            save_reconstruction_figure(
                directory / "reconstruction.png",
                target_map,
                prediction_map,
                valid_map,
            )
        print(
            f"[{index + 1}/{len(dataset)}] scene={record.scene_id} "
            f"rmse={score['rmse']:.4f} nmse={score['nmse']:.6f}"
        )

    with (result_root / "metrics_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for metric in (*METRIC_NAMES, *(f"raw_{name}" for name in METRIC_NAMES)):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary.update({
        "maps": len(rows),
        "unique_scenes": len({int(row["scene_id"]) for row in rows}),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "primary_metric_domain": "normalized_minus1_to_1",
        "evaluation_support": "unobserved_valid_rx_only" if float(sparse.get("sample_rate", 0.0)) > 0 else "valid_rx_only",
        "sparse_sample_rate": float(sparse.get("sample_rate", 0.0)),
        "normalization": "x / 127.5 - 1.0",
        "raw_metric_prefix": "raw_",
    })
    (result_root / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (result_root / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std"))
        writer.writeheader()
        for metric in (*METRIC_NAMES, *(f"raw_{name}" for name in METRIC_NAMES)):
            writer.writerow({"metric": metric, **summary[metric]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
