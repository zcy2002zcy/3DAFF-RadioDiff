"""Audit the PEFNet-inspired baseline information contract and forward pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dataset import PathlossDataset
from model import AdaptedPEFNet


DATA_ROOT = Path(r"D:\datas\processed_pl_aff_dataset")
BUILDING_DIR = Path(r"D:\zdata\Building_Infomation\buildings")


def main() -> None:
    dataset = PathlossDataset(
        DATA_ROOT,
        "test",
        BUILDING_DIR,
        sampling_rate=0.10,
        sampling_mode="random",
        limit=1,
    )
    sample = dataset[0]
    mask = sample["mask"].numpy().astype(bool)
    sparse = sample["sparse"].numpy()
    if np.any(sparse[~mask] != 0):
        raise AssertionError("Unobserved target values leaked into sparse input")
    expected = round(256 * 256 * 0.10)
    if int(mask.sum()) != expected:
        raise AssertionError(f"Expected {expected} samples, found {int(mask.sum())}")

    model = AdaptedPEFNet(base_channels=32).eval()
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }
    with torch.no_grad():
        output = model(batch)["prediction"]
    if output.shape != (1, 1, 256, 256):
        raise AssertionError(f"Unexpected output shape: {tuple(output.shape)}")

    source_file = dataset.files[0][1]
    with np.load(source_file, allow_pickle=False) as raw:
        source_keys = list(raw.files)
    report = {
        "status": "PASS",
        "method_label": "PEFNet-inspired building-input adaptation",
        "authorized_model_inputs": [
            "q",
            "raw building polygons/heights",
            "10% sparse path loss",
            "sampling mask",
            "receiver altitude z",
        ],
        "ignored_npz_fields": ["los", "depth"],
        "source_npz_keys": source_keys,
        "sample_count": int(mask.sum()),
        "sample_fraction": float(mask.mean()),
        "output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "target_use": "training loss, sampled values, and metrics only",
    }
    Path("pefnet_building_io_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
