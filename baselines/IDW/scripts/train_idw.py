import argparse
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    with open(root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt_dir = root / cfg["checkpoint"]["save_dir"]
    log_dir = root / cfg["logging"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    marker = {"method": "idw", "config": cfg, "note": "IDW is a non-parametric baseline; no training is required."}
    (ckpt_dir / "best.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    print("IDW is non-parametric. Training step completed by writing marker:")
    print(ckpt_dir / "best.json")


if __name__ == "__main__":
    main()
