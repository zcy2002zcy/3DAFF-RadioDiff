import argparse
import sys
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.radiomap_dataset import build_dataloader_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("=" * 80)
    print("IDW PATHLOSS PRE-RUN CHECK")
    print("Config:", cfg_path)
    print("=" * 80)
    print("task_name:", cfg.get("task_name"))
    print("method input: sparse measurements only")
    print("sampling_mode:", cfg["data"].get("sampling_mode"))
    print("sample_rate:", cfg["data"].get("sample_rate"))
    print("k_neighbors:", cfg["model"].get("k_neighbors"))
    for key in ["train_dir", "val_dir", "test_dir"]:
        p = Path(cfg["data"][key])
        print(f"{key}: {p} | exists={p.exists()} | npz={len(list(p.glob('*.npz'))) if p.exists() else 0}")
    loader = build_dataloader_from_config(cfg["data"]["test_dir"], cfg, shuffle=False, drop_last=False, split="test")
    batch = next(iter(loader))
    forbidden = {"l", "los", "depth"}.intersection(batch)
    assert not forbidden, f"Forbidden derived geometry entered baseline batch: {forbidden}"
    assert torch.count_nonzero(batch["s"] * (1.0 - batch["s_mask"])).item() == 0
    print("q shape:", tuple(batch["q"].shape))
    print("sampled shape:", tuple(batch["s"].shape))
    print("sample mask shape:", tuple(batch["s_mask"].shape))
    print("target shape:", tuple(batch["psi"].shape))
    print(f"actual_sample_fraction={batch['s_mask'].float().mean().item():.6f}")
    try:
        import scipy  # noqa: F401
        print("scipy: available")
    except Exception:
        print("scipy: not available, eval will use slower numpy fallback")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
