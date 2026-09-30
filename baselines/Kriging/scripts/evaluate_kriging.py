import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metrics import compute_nmse, compute_psnr, compute_rmse, compute_ssim, gaussian_window
from scripts.evaluation_protocol import evaluate_regions


METRIC_KEYS = [
    "loss", "nmse", "rmse", "ssim", "psnr",
    "all_loss", "all_nmse", "all_rmse", "all_ssim", "all_psnr",
    "valid_loss", "valid_nmse", "valid_rmse", "valid_ssim", "valid_psnr",
    "valid_fraction", "eval_fraction",
]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def random_observations(target, sample_rate, seed):
    total = target.size
    count = max(1, int(round(total * sample_rate)))
    indices = np.random.default_rng(seed).choice(total, size=count, replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[indices] = True
    mask = mask.reshape(target.shape)
    sparse = np.zeros_like(target, dtype=np.float32)
    sparse[mask] = target[mask]
    yy, xx = np.nonzero(mask)
    points = np.column_stack([yy, xx]).astype(np.float64)
    values = target[mask].astype(np.float64)
    return sparse, mask, points, values


@lru_cache(maxsize=128)
def rasterize_building_mask(json_path, image_size):
    objects = json.loads(Path(json_path).read_text(encoding="utf-8"))
    image = Image.new("F", (image_size, image_size), 0.0)
    draw = ImageDraw.Draw(image)
    for vertices, _ in objects:
        polygon = [(float(x) - 1.0, float(image_size) - float(y)) for x, y in vertices]
        draw.polygon(polygon, fill=1.0)
    return np.asarray(image, dtype=np.float32)


def query_grid(shape):
    yy, xx = np.indices(shape)
    return np.column_stack([yy.reshape(-1), xx.reshape(-1)]).astype(np.float64)


def ordinary_kriging(points, values, queries, variogram_model, neighbors, chunk_size):
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise RuntimeError("PyKrige is required. Install pykrige; no fallback is permitted.") from exc

    if len(points) < 3:
        raise ValueError("Ordinary Kriging requires at least three observations")
    model = OrdinaryKriging(
        points[:, 1],
        points[:, 0],
        values,
        variogram_model=variogram_model,
        exact_values=True,
        pseudo_inv=True,
        verbose=False,
        enable_plotting=False,
    )
    predictions = np.empty(len(queries), dtype=np.float64)
    local_count = min(int(neighbors), len(points))
    for start in range(0, len(queries), int(chunk_size)):
        chunk = queries[start : start + int(chunk_size)]
        values_chunk, _ = model.execute(
            "points",
            chunk[:, 1],
            chunk[:, 0],
            backend="loop",
            n_closest_points=local_count,
        )
        predictions[start : start + len(chunk)] = np.asarray(values_chunk)
    return predictions


def topology_kriging(points, values, queries, building_mask, variogram_model, neighbors, chunk_size):
    prediction = ordinary_kriging(
        points, values, queries, variogram_model, neighbors, chunk_size
    )
    sample_class = building_mask[points[:, 0].astype(int), points[:, 1].astype(int)] >= 0.5
    query_class = building_mask[queries[:, 0].astype(int), queries[:, 1].astype(int)] >= 0.5
    for is_building in (False, True):
        sample_selector = sample_class == is_building
        query_selector = query_class == is_building
        if np.count_nonzero(sample_selector) < 3 or not np.any(query_selector):
            continue
        prediction[query_selector] = ordinary_kriging(
            points[sample_selector],
            values[sample_selector],
            queries[query_selector],
            variogram_model,
            neighbors,
            chunk_size,
        )
    return prediction


def metric_values(prediction, target, building_mask, sample_mask, window):
    pred = torch.from_numpy(prediction[None, None]).float()
    truth = torch.from_numpy(target[None, None]).float()
    buildings = torch.from_numpy(building_mask[None, None]).float()
    samples = torch.from_numpy(sample_mask[None, None].astype(np.float32))
    metrics = evaluate_regions(pred, truth, buildings, samples, window)
    return {key: float(metrics[key][0]) for key in METRIC_KEYS}


def summarize(records, method):
    result = {"method": method, "count": len(records)}
    for key in METRIC_KEYS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        result[key] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{key}_ci95"] = float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=2025)
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    test_dir = Path(cfg["data"]["test_dir"])
    building_dir = Path(cfg["data"]["building_dir"])
    all_files = sorted(test_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No test NPZ files in {test_dir}")
    indexed_files = list(enumerate(all_files))
    if args.max_samples > 0 and args.max_samples < len(indexed_files):
        selected = np.random.default_rng(args.subset_seed).choice(
            len(indexed_files), size=args.max_samples, replace=False
        )
        indexed_files = [indexed_files[index] for index in sorted(selected.tolist())]

    output_root = resolve_path(cfg["eval"]["output_dir"])
    save_first_n = int(cfg["eval"].get("save_first_n", 20))
    sample_rate = float(cfg["data"].get("sample_rate", 0.10))
    variogram = str(cfg["model"].get("variogram_model", "linear"))
    neighbors = int(cfg["model"].get("n_closest_points", 32))
    chunk_size = int(cfg["model"].get("query_chunk_size", 4096))
    window = gaussian_window(channels=1, device="cpu", dtype=torch.float32)
    queries = query_grid((int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"])))

    print(f"Test samples: {len(indexed_files)}")
    print(f"Sampling: random {sample_rate:.6f}; exact count={round(queries.shape[0] * sample_rate)}")
    print("Ordinary input: sparse measurements only")
    print("Topology input: sparse measurements + raw building mask")
    print("Backend: PyKrige only; local moving window")

    all_summaries = []
    for method in ("kriging", "kriging_building_topology"):
        method_dir = output_root / method
        records = []
        for order, (source_index, path) in enumerate(tqdm(indexed_files, desc=method)):
            with np.load(path, allow_pickle=True) as data:
                target = data["psi"].astype(np.float32)
                sample_id = str(np.asarray(data["id"]).item())
                q = data["q"].astype(np.float32)
            # The other standardized evaluators seed each fixed test mask by the
            # sample's index in the complete sorted 40,000-file split.
            sparse, mask, points, values = random_observations(target, sample_rate, source_index)
            building_id = sample_id.split("_", 1)[0]
            building_mask = rasterize_building_mask(
                str(building_dir / f"{building_id}.json"), int(cfg["data"]["image_size"])
            )
            if method == "kriging":
                flat_prediction = ordinary_kriging(
                    points, values, queries, variogram, neighbors, chunk_size
                )
            else:
                flat_prediction = topology_kriging(
                    points, values, queries, building_mask, variogram, neighbors, chunk_size
                )
            prediction = flat_prediction.reshape(target.shape).astype(np.float32)
            prediction = np.clip(prediction, -1.0, 1.0)
            prediction[mask] = sparse[mask]
            metrics = metric_values(prediction, target, building_mask, mask, window)
            records.append({"index": order, "source_index": source_index, "id": sample_id, **metrics})
            if order < save_first_n:
                method_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    method_dir / f"sample_{order:04d}.npz",
                    prediction=prediction,
                    target=target,
                    q=q,
                    building_mask=building_mask,
                    sampled=sparse,
                    sample_mask=mask.astype(np.float32),
                )
        summary = summarize(records, method)
        write_csv(method_dir / "metrics_per_sample.csv", records)
        write_csv(method_dir / "metrics_summary.csv", [summary])
        all_summaries.append(summary)
        print(summary)
    write_csv(output_root / "metrics_summary_all.csv", all_summaries)


if __name__ == "__main__":
    main()
