import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.resnet import ResNetBaseline


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_input(batch):
    height, width = batch["q"].shape[-2:]
    altitude = ((batch["z"].view(-1, 1, 1, 1) - 1.0) / 3.0).clamp(0.0, 1.0)
    altitude_map = altitude.expand(-1, -1, height, width)
    return torch.cat(
        [batch["q"], batch["building_mask"], batch["s"], altitude_map],
        dim=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_yaml(cfg_path)

    print("=" * 80)
    print("RESNET PATHLOSS PRE-RUN CHECK")
    print("Config:", cfg_path)
    print("=" * 80)
    print("task_name:", cfg.get("task_name"))
    print("sampling_mode:", cfg["data"].get("sampling_mode"))
    print("sample_rate:", cfg["data"].get("sample_rate"))
    print("condition:", "q + binary building mask + sparse measurements + altitude")
    print("excluded:", "NPZ depth, NPZ LoS, and building-height values")
    print("batch_size:", cfg["train"].get("batch_size"))

    for key in ["train_dir", "val_dir", "test_dir"]:
        path = Path(cfg["data"][key])
        files = list(path.glob("*.npz")) if path.exists() else []
        print(f"{key}: {path} | exists={path.exists()} | npz={len(files)}")
        if files:
            import numpy as np
            with np.load(files[0], allow_pickle=True) as data:
                print("first keys:", list(data.files))
                for k in ["q", "psi"]:
                    if k in data.files:
                        print(f"{k} shape:", data[k].shape)

    loader = build_dataloader_from_config(cfg["data"]["train_dir"], cfg, shuffle=False, drop_last=True, split="train")
    batch = next(iter(loader))
    forbidden = {"l", "los", "depth"}.intersection(batch)
    assert not forbidden, f"Forbidden derived geometry entered baseline batch: {forbidden}"
    assert torch.count_nonzero(batch["s"] * (1.0 - batch["s_mask"])).item() == 0
    x = build_input(batch)
    y = batch["psi"]
    model_cfg = cfg.get("model", {})
    model = ResNetBaseline(
        in_channels=int(model_cfg.get("in_channels", x.shape[1])),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 22)),
        blocks_per_stage=int(model_cfg.get("blocks_per_stage", 2)),
    )
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"built model: ResNetBaseline | params={params:.2f}M")
    print(f"input_shape={tuple(x.shape)} target_shape={tuple(y.shape)} out_shape={tuple(out.shape)}")
    print(f"actual_sample_fraction={batch['s_mask'].float().mean().item():.6f}")
    print("=" * 80)
    print("CHECK PASSED")


if __name__ == "__main__":
    main()



