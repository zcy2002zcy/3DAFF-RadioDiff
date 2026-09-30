import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import RadioDiff3DVolumeDataset
from diffusion import GaussianDiffusion
from utils.common import build_condition, build_model, load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    manifest_path = project_path(data["manifest_dir"]) / "test.json"
    dataset = RadioDiff3DVolumeDataset(
        str(manifest_path), building_dir=data["building_dir"],
        image_size=data["image_size"], max_building_height=data["max_building_height"],
        use_sampling=data["use_sampling"], sample_rate=data["sample_rate"],
        deterministic_sampling=True, seed=cfg["project"]["seed"], max_volumes=1
    )
    sample = dataset[0]
    batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in sample.items()}
    condition = build_condition(batch, cfg)
    model = build_model(cfg).eval()
    smoke_target = batch["psi"][..., :64, :64]
    smoke_condition = condition[..., :64, :64]
    with torch.no_grad():
        output = model(smoke_target, smoke_condition, torch.tensor([100]))
    dcfg = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=dcfg["timesteps"], schedule=dcfg["beta_schedule"],
        beta_start=dcfg["beta_start"], beta_end=dcfg["beta_end"],
        cosine_s=dcfg.get("cosine_s", 0.008),
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print("=" * 78)
    print("RadioDiff-3D PL / q + building information check")
    print(f"volume id: {sample['id']}")
    print(f"psi: {tuple(batch['psi'].shape)}")
    suffix = " + sparse_pl" if data["use_sampling"] else ""
    print(f"condition: {tuple(condition.shape)} = q + building_mask + building_height{suffix}")
    print(f"random sample fraction: {float(batch['s_mask'].mean()):.6f}")
    print(f"model smoke output: {tuple(output.shape)} (64x64 crop)")
    print(f"parameters: {parameter_count / 1e6:.2f} M")
    print(f"training T: {diffusion.timesteps}")
    print(f"terminal alpha_bar: {float(diffusion.alpha_bar[-1]):.10f}")
    print(f"DDIM inference steps: {cfg['diffusion']['ddim_steps']}")
    print("LoS/AFF-depth condition used: False")
    print("AFF used: False")
    print("CHECK PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
