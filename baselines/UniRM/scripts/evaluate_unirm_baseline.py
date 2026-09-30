"""Evaluate UniRM adapted with the common scene-disjoint test protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dense_baseline_dataset import DenseBaselineDataset
from models.unirm_baseline import UPSTREAM_COMMIT, build_unirm_adapted
from scripts.evaluate import metrics, normalized_metrics


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-maps", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    name = f"unirm_adapted_random0_seed{args.seed}" + ("_smoke" if args.smoke_test else "")
    output = PROJECT_ROOT / "outputs" / name
    checkpoint_path = args.checkpoint or output / "checkpoints" / "best.pt"
    dataset = DenseBaselineDataset(
        PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv",
        max_records=2 if args.smoke_test else args.max_maps,
        selection_strategy="sequential" if args.smoke_test else "balanced_scene",
        seed=args.data_seed, sample_rate=0.0, training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unirm_adapted(enable_prompt=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    rows = []
    for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
        prediction = model(batch["input"].to(device)).cpu().numpy()[0, 0] * 255.0
        target = batch["target_raw"].numpy()[0, 0]
        valid = batch["valid_mask"].numpy()[0, 0].astype(bool)
        score = normalized_metrics(prediction[valid], target[valid])
        raw = metrics(prediction[valid], target[valid], 255.0)
        rows.append({
            "map_index": index, "scene_id": int(batch["scene_id"]),
            "valid_points": int(valid.sum()), **score,
            **{f"raw_{key}": value for key, value in raw.items()},
        })
        print(f"[{index + 1}/{len(dataset)}] rmse={score['rmse']:.4f} nmse={score['nmse']:.6f}")
    result = output / "evaluation" / "test"
    result.mkdir(parents=True, exist_ok=True)
    with (result / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    metric_names = ("rmse", "nmse", "mae", "psnr", "raw_rmse", "raw_nmse", "raw_mae", "raw_psnr")
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary.update({
        "maps": len(rows), "seed": args.seed, "data_seed": args.data_seed, "support": "valid_rx_only",
        "primary_metric_domain": "normalized_minus1_to_1",
        "normalization": "x / 127.5 - 1.0",
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_stage": checkpoint["stage"],
        "method_label": "UniRM (adapted, no-sampling)",
        "upstream_commit": UPSTREAM_COMMIT,
        "external_pretraining": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    })
    (result / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
