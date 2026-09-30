"""Evaluate adapted PEFNet with the same four metrics used by 3D-RadioDiff."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aligned_metrics import METRIC_NAMES, aligned_batch_metrics
from dataset import PathlossDataset
from model import AdaptedPEFNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--building-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage", choices=("physics", "refine", "joint"), default=None)
    parser.add_argument("--sampling-rate", type=float, default=None)
    parser.add_argument("--sampling-mode", choices=("random", "uniform"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def setting(cli: argparse.Namespace, saved: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(cli, name)
    return value if value is not None else saved.get(name, default)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def write_results(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "split", "count", *METRIC_NAMES))
        writer.writeheader()
        writer.writerow(summary)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


@torch.no_grad()
def evaluate(
    model: AdaptedPEFNet,
    loader: DataLoader,
    device: torch.device,
    stage: str,
    amp: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    rows: list[dict[str, Any]] = []
    prediction_key = "coarse" if stage == "physics" else "prediction"

    for batch in tqdm(loader, desc="eval:test"):
        batch = move_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(batch)
        prediction = outputs[prediction_key].clamp(0.0, 1.0)
        prediction = torch.where(batch["mask"].bool(), batch["sparse"], prediction)
        batch_metrics = aligned_batch_metrics(prediction, batch["target"])
        ids = list(batch["id"])
        metric_values = {name: values.detach().cpu().tolist() for name, values in batch_metrics.items()}
        for index, sample_id in enumerate(ids):
            row = {"id": sample_id}
            for name in METRIC_NAMES:
                value = float(metric_values[name][index])
                row[name] = value
                totals[name] += value
            rows.append(row)

    count = len(rows)
    summary = {"model": "PEFNet-adapted", "split": "test", "count": count}
    summary.update({name: totals[name] / max(count, 1) for name in METRIC_NAMES})
    return rows, summary


def main() -> None:
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload

    seed = int(setting(cli, saved, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = setting(cli, saved, "device", "cuda")
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    stage = setting(cli, saved, "stage", "refine")
    amp = bool(setting(cli, saved, "amp", True))
    num_workers = int(setting(cli, saved, "num_workers", 4))

    dataset = PathlossDataset(
        root=setting(cli, saved, "data_root", r"D:\datas\processed_pl_aff_dataset"),
        split="test",
        building_dir=setting(
            cli, saved, "building_dir", r"D:\zdata\Building_Infomation\buildings"
        ),
        sampling_rate=float(setting(cli, saved, "sampling_rate", 0.10)),
        sampling_mode=setting(cli, saved, "sampling_mode", "random"),
        max_height=float(saved.get("max_height", 20.0)),
        max_building_height=float(saved.get("max_building_height", 20.0)),
        subset_seed=seed,
        limit=cli.test_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(setting(cli, saved, "batch_size", 2)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = AdaptedPEFNet(
        image_size=int(saved.get("image_size", 256)),
        base_channels=int(saved.get("base_channels", 32)),
        frequency_mhz=float(saved.get("frequency_mhz", 3500.0)),
        pixel_size_m=float(saved.get("pixel_size_m", 1.0)),
        contrast_scale=float(saved.get("contrast_scale", 1.0)),
        coupling=float(saved.get("vie_coupling", 0.25)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    rows, summary = evaluate(model, loader, device, stage, amp)
    output_dir = Path(cli.output_dir) if cli.output_dir else checkpoint_path.parent / "eval_test"
    write_results(output_dir, rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
