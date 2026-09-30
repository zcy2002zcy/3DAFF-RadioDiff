"""Four-altitude RadioDiff-3D data with geometry-only building conditions."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm


def _scalar(value):
    array = np.asarray(value)
    return array.item() if array.size == 1 else array.tolist()


def _read_identity(path: str) -> Tuple[str, float, str]:
    with np.load(path, allow_pickle=True) as data:
        return str(_scalar(data["id"])), float(_scalar(data["z"])), path


def build_manifest(data_dir: str, output_path: str, altitudes: Sequence[float], workers: int = 8):
    files = sorted(str(path) for path in Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")
    file_indices = {path: index for index, path in enumerate(files)}
    groups = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = pool.map(_read_identity, files, chunksize=64)
        for sample_id, altitude, path in tqdm(rows, total=len(files), desc="Index NPZ"):
            groups[sample_id][altitude].append(path)
    expected = tuple(float(value) for value in altitudes)
    volumes, invalid = [], []
    for sample_id in sorted(groups):
        by_altitude = groups[sample_id]
        missing = [value for value in expected if value not in by_altitude]
        counts = [len(by_altitude[value]) for value in expected if value in by_altitude]
        if missing or len(counts) != len(expected) or len(set(counts)) != 1:
            invalid.append(f"{sample_id}: missing={missing}, counts={counts}")
            continue
        for occurrence in range(counts[0]):
            volumes.append({
                "id": sample_id,
                "occurrence": occurrence,
                "files": [by_altitude[value][occurrence] for value in expected],
                "file_indices": [
                    file_indices[by_altitude[value][occurrence]] for value in expected
                ],
            })
    if invalid:
        print(f"Warning: skipped {len(invalid)} incomplete altitude groups")
        print("\n".join(invalid[:20]))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "data_dir": str(Path(data_dir).resolve()),
        "altitudes": list(expected),
        "file_count": len(files),
        "volume_count": len(volumes),
        "volumes": volumes,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(volumes)} volumes to {output}")
    return volumes


def build_development_manifests(
    train_dir: str,
    val_dir: str,
    train_output: str,
    val_output: str,
    altitudes: Sequence[float],
    val_fraction: float,
    seed: int,
    workers: int = 8,
):
    """Recover complete volumes from train+val, then split by scene ID."""
    files = sorted(
        [str(path) for directory in (train_dir, val_dir) for path in Path(directory).glob("*.npz")]
    )
    groups = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = pool.map(_read_identity, files, chunksize=64)
        for sample_id, altitude, path in tqdm(rows, total=len(files), desc="Index train+val NPZ"):
            groups[sample_id][altitude].append(path)
    expected = tuple(float(value) for value in altitudes)
    by_id = {}
    incomplete = []
    for sample_id in sorted(groups):
        slices = groups[sample_id]
        missing = [value for value in expected if value not in slices]
        counts = [len(slices[value]) for value in expected if value in slices]
        if missing or len(counts) != len(expected) or len(set(counts)) != 1:
            incomplete.append(sample_id)
            continue
        by_id[sample_id] = [{
            "id": sample_id,
            "occurrence": occurrence,
            "files": [slices[value][occurrence] for value in expected],
        } for occurrence in range(counts[0])]
    if incomplete:
        print(
            f"Warning: excluded {len(incomplete)} incomplete development scenes; "
            f"examples={incomplete[:20]}"
        )

    scene_ids = np.asarray(sorted(by_id), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(scene_ids)
    val_count = max(1, int(round(len(scene_ids) * val_fraction)))
    val_ids = set(scene_ids[:val_count].tolist())
    payloads = {
        "train": [item for sample_id in scene_ids if sample_id not in val_ids for item in by_id[sample_id]],
        "val": [item for sample_id in scene_ids if sample_id in val_ids for item in by_id[sample_id]],
    }
    for split, output_path in (("train", train_output), ("val", val_output)):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "source_dirs": [str(Path(train_dir).resolve()), str(Path(val_dir).resolve())],
            "split_policy": "scene_id",
            "seed": seed,
            "val_fraction": val_fraction,
            "excluded_incomplete_scene_count": len(incomplete),
            "excluded_incomplete_scene_examples": incomplete[:20],
            "altitudes": list(expected),
            "volume_count": len(payloads[split]),
            "volumes": payloads[split],
        }, indent=2), encoding="utf-8")
        print(f"Saved {len(payloads[split])} {split} volumes to {output}")
    return payloads


def load_manifest(path: str):
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}. Run run_prepare.ps1 first.")
    return json.loads(manifest.read_text(encoding="utf-8"))["volumes"]


@lru_cache(maxsize=128)
def rasterize_buildings(json_path: str, image_size: int, max_height: float):
    """Render building maps using audited (x, y) -> (x-1, H-y) coordinates."""
    objects = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mask_image = Image.new("F", (image_size, image_size), 0.0)
    height_image = Image.new("F", (image_size, image_size), 0.0)
    mask_draw, height_draw = ImageDraw.Draw(mask_image), ImageDraw.Draw(height_image)
    for vertices, height_container in objects:
        height_value = float(np.asarray(height_container).reshape(-1)[0])
        polygon = [(float(x) - 1.0, float(image_size) - float(y)) for x, y in vertices]
        mask_draw.polygon(polygon, fill=1.0)
        height_draw.polygon(polygon, fill=min(max(height_value / max_height, 0.0), 1.0))
    return np.asarray(mask_image, dtype=np.float32), np.asarray(height_image, dtype=np.float32)


class RadioDiff3DVolumeDataset(Dataset):
    """Return paper-defined q, building mask/height, and optional sparse PL."""

    def __init__(self, manifest_path: str, building_dir: str, image_size: int = 256,
                 max_building_height: float = 20.0, use_sampling: bool = True,
                 sample_rate: float = 0.10, deterministic_sampling: bool = False,
                 seed: int = 42, max_volumes: int = 0) -> None:
        self.volumes = load_manifest(manifest_path)
        if max_volumes > 0:
            self.volumes = self.volumes[:max_volumes]
        self.building_dir = Path(building_dir)
        self.image_size = int(image_size)
        self.max_building_height = float(max_building_height)
        self.use_sampling = bool(use_sampling)
        self.sample_rate = float(sample_rate)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.seed = int(seed)
        self._rng: Optional[np.random.Generator] = None

    def __len__(self):
        return len(self.volumes)

    def _generator(self, index: int):
        if self.deterministic_sampling:
            return np.random.default_rng(self.seed + index)
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = torch.initial_seed() if worker is not None else self.seed
            self._rng = np.random.default_rng(worker_seed % (2**32))
        return self._rng

    def __getitem__(self, index: int):
        item = self.volumes[index]
        targets, transmitter_maps, altitudes = [], [], []
        for path in item["files"]:
            with np.load(path, allow_pickle=True) as data:
                targets.append(data["psi"].astype(np.float32))
                transmitter_maps.append(data["q"].astype(np.float32))
                altitudes.append(float(_scalar(data["z"])))
        psi, q = np.stack(targets), np.stack(transmitter_maps)
        building_id = str(item["id"]).split("_", 1)[0]
        geometry_path = self.building_dir / f"{building_id}.json"
        if not geometry_path.exists():
            raise FileNotFoundError(f"Building geometry not found: {geometry_path}")
        mask_2d, height_2d = rasterize_buildings(
            str(geometry_path), self.image_size, self.max_building_height
        )
        building_mask = np.repeat(mask_2d[None], psi.shape[0], axis=0)
        building_height = np.repeat(height_2d[None], psi.shape[0], axis=0)
        sparse, sparse_mask = np.zeros_like(psi), np.zeros_like(psi)
        if self.use_sampling:
            count = max(1, int(round(psi.shape[-2] * psi.shape[-1] * self.sample_rate)))
            for depth_index in range(psi.shape[0]):
                if self.deterministic_sampling and "file_indices" in item:
                    # Match the main 2D dataset exactly: seed = sorted NPZ file index.
                    rng = np.random.default_rng(int(item["file_indices"][depth_index]))
                else:
                    rng = self._generator(index)
                chosen = rng.choice(psi.shape[-2] * psi.shape[-1], count, replace=False)
                sparse[depth_index].reshape(-1)[chosen] = psi[depth_index].reshape(-1)[chosen]
                sparse_mask[depth_index].reshape(-1)[chosen] = 1.0
        if np.any(sparse[sparse_mask == 0] != 0):
            raise RuntimeError("Sparse target leakage outside the sampling mask")

        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
        return {
            "id": str(item["id"]), "psi": tensor(psi), "q": tensor(q),
            "building_mask": tensor(building_mask),
            "building_height": tensor(building_height),
            "s": tensor(sparse), "s_mask": tensor(sparse_mask),
            "z": torch.tensor(altitudes, dtype=torch.float32),
        }
