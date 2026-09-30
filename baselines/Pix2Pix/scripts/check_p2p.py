import argparse
import sys
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config
from models.pix2pix import PatchDiscriminator, Pix2PixGenerator


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
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_yaml(cfg_path)
    print("=" * 80)
    print("PIX2PIX PATHLOSS PRE-RUN CHECK")
    print("Config:", cfg_path)
    print("=" * 80)
    print("task_name:", cfg.get("task_name"))
    print("sampling_mode:", cfg["data"].get("sampling_mode"))
    print("sample_rate:", cfg["data"].get("sample_rate"))
    print("condition:", "q + binary building mask + sparse measurements + altitude")
    print("excluded:", "NPZ depth, NPZ LoS, and building-height values")
    print("batch_size:", cfg["train"].get("batch_size"))
    for key in ["train_dir", "val_dir", "test_dir"]:
        p = Path(cfg["data"][key])
        print(f"{key}: {p} | exists={p.exists()} | npz={len(list(p.glob('*.npz'))) if p.exists() else 0}")
    loader = build_dataloader_from_config(cfg["data"]["train_dir"], cfg, shuffle=False, drop_last=True, split="train")
    batch = next(iter(loader))
    forbidden = {"l", "los", "depth"}.intersection(batch)
    assert not forbidden, f"Forbidden derived geometry entered baseline batch: {forbidden}"
    assert torch.count_nonzero(batch["s"] * (1.0 - batch["s_mask"])).item() == 0
    x = build_input(batch)
    y = batch["psi"]
    mcfg = cfg["model"]
    gen = Pix2PixGenerator(mcfg["in_channels"], mcfg["out_channels"], mcfg["base_channels"])
    disc = PatchDiscriminator(mcfg["in_channels"] + mcfg["out_channels"], mcfg["base_channels"])
    with torch.no_grad():
        pred = gen(x)
        logits = disc(x, pred)
    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"generator_params={gen_params:.3f}M discriminator_params={disc_params:.3f}M")
    print(f"input_shape={tuple(x.shape)} target_shape={tuple(y.shape)} pred_shape={tuple(pred.shape)} disc_shape={tuple(logits.shape)}")
    print(f"actual_sample_fraction={batch['s_mask'].float().mean().item():.6f}")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
