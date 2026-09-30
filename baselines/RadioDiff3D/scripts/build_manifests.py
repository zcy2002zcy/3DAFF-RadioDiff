import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import build_development_manifests, build_manifest
from utils.common import load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild manifests even when the output JSON already exists.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    manifest_dir = project_path(data["manifest_dir"])
    train_output = manifest_dir / "train.json"
    val_output = manifest_dir / "val.json"
    if args.force or not train_output.exists() or not val_output.exists():
        build_development_manifests(
            train_dir=data["train_dir"], val_dir=data["val_dir"],
            train_output=str(train_output), val_output=str(val_output),
            altitudes=data["altitudes"], val_fraction=data["val_fraction"],
            seed=cfg["project"]["seed"], workers=args.workers,
        )
    else:
        print(f"Skip existing manifests: {train_output}, {val_output}")

    test_output = manifest_dir / "test.json"
    if args.force or not test_output.exists():
        build_manifest(
            data_dir=data["test_dir"], output_path=str(test_output),
            altitudes=data["altitudes"], workers=args.workers,
        )
    else:
        print(f"Skip existing manifest: {test_output}")


if __name__ == "__main__":
    main()
