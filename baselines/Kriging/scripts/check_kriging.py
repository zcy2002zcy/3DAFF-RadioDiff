import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_kriging import random_observations, rasterize_building_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert importlib.util.find_spec("pykrige") is not None, "PyKrige is required"
    files = sorted(Path(cfg["data"]["test_dir"]).glob("*.npz"))
    assert files, "No test files found"
    with np.load(files[0], allow_pickle=True) as data:
        assert "psi" in data.files and "q" in data.files
        target = data["psi"].astype(np.float32)
        sample_id = str(np.asarray(data["id"]).item())
    sparse, mask, points, values = random_observations(target, 0.10, 42)
    building_path = Path(cfg["data"]["building_dir"]) / f"{sample_id.split('_', 1)[0]}.json"
    building_mask = rasterize_building_mask(str(building_path), target.shape[0])
    assert len(points) == round(target.size * 0.10)
    assert np.count_nonzero(sparse[~mask]) == 0
    print("PyKrige: available")
    print("test_count:", len(files))
    print("sample_count:", len(points))
    print("sample_fraction:", mask.mean())
    print("building_mask_shape:", building_mask.shape)
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
