"""Require scene-level isolation across train, validation, and test manifests."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="cache/manifests")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(args.manifest_dir)
    if not directory.is_absolute():
        directory = root / directory
    ids = {}
    for split in ("train", "val", "test"):
        path = directory / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids[split] = {str(item["id"]) for item in payload["volumes"]}
    overlaps = {
        "train_val": ids["train"] & ids["val"],
        "train_test": ids["train"] & ids["test"],
        "val_test": ids["val"] & ids["test"],
    }
    for name, values in overlaps.items():
        print(f"{name} overlap: {len(values)} | examples={sorted(values)[:10]}")
    if any(overlaps.values()):
        raise SystemExit("AUDIT FAILED: split scenes are not independent.")
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
