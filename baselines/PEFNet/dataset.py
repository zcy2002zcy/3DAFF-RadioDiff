"""Leakage-safe path-loss adapter for the building-input PEFNet baseline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


@lru_cache(maxsize=128)
def rasterize_buildings(
    json_path: str,
    image_size: int,
    max_building_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize raw Urban3D polygons using the dataset's x/y convention."""
    objects = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mask_image = Image.new("F", (image_size, image_size), 0.0)
    height_image = Image.new("F", (image_size, image_size), 0.0)
    mask_draw = ImageDraw.Draw(mask_image)
    height_draw = ImageDraw.Draw(height_image)
    for vertices, height_container in objects:
        building_height = float(np.asarray(height_container).reshape(-1)[0])
        polygon = [
            (float(x) - 1.0, float(image_size) - float(y))
            for x, y in vertices
        ]
        mask_draw.polygon(polygon, fill=1.0)
        normalized = np.clip(building_height / max_building_height, 0.0, 1.0)
        height_draw.polygon(polygon, fill=float(normalized))
    return (
        np.asarray(mask_image, dtype=np.float32),
        np.asarray(height_image, dtype=np.float32),
    )


class PathlossDataset(Dataset):
    """Return only q, raw-building derivatives, sparse PL, z, and the target.

    NPZ ``los`` and ``depth`` are deliberately ignored. ``psi`` is used only
    as the supervised target and to populate values selected by the mask.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        building_dir: str | Path,
        sampling_rate: float = 0.10,
        sampling_mode: str = "random",
        max_height: float = 20.0,
        max_building_height: float = 20.0,
        subset_seed: int = 42,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.building_dir = Path(building_dir)
        self.sampling_rate = float(sampling_rate)
        self.sampling_mode = sampling_mode.lower()
        self.max_height = float(max_height)
        self.max_building_height = float(max_building_height)

        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1]")
        if self.sampling_mode not in {"random", "uniform"}:
            raise ValueError("sampling_mode must be random or uniform")
        if self.max_height <= 0 or self.max_building_height <= 0:
            raise ValueError("height scales must be positive")
        if not self.building_dir.is_dir():
            raise FileNotFoundError(f"Missing building directory: {self.building_dir}")

        split_dir = self.root / split
        all_files = sorted(split_dir.glob("*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No NPZ samples found in {split_dir}")
        self.files = list(enumerate(all_files))
        if limit is not None and limit > 0:
            rng = np.random.default_rng(subset_seed + sum(map(ord, split)))
            chosen = rng.choice(len(self.files), min(limit, len(self.files)), replace=False)
            self.files = [self.files[int(i)] for i in sorted(chosen)]

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _map(array: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32).squeeze()
        if value.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got {value.shape}")
        return value

    @staticmethod
    def _zero_one(psi: np.ndarray) -> np.ndarray:
        psi = PathlossDataset._map(psi, "psi")
        if float(np.nanmin(psi)) < 0.0:
            psi = (psi + 1.0) * 0.5
        return np.nan_to_num(psi, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    def _sample(self, target: np.ndarray, original_index: int) -> Tuple[np.ndarray, np.ndarray]:
        height, width = target.shape
        count = max(1, int(round(target.size * self.sampling_rate)))
        if self.sampling_mode == "random":
            seed = original_index if self.split in {"val", "test"} else None
            indices = np.random.default_rng(seed).choice(target.size, count, replace=False)
        else:
            rows_n = max(1, int(round(np.sqrt(count * height / width))))
            cols_n = max(1, int(round(count / rows_n)))
            rows = np.linspace(0, height - 1, rows_n, dtype=np.int64)
            cols = np.linspace(0, width - 1, cols_n, dtype=np.int64)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            indices = np.unique(rr.reshape(-1) * width + cc.reshape(-1))

        sparse = np.zeros_like(target, dtype=np.float32)
        mask = np.zeros_like(target, dtype=np.float32)
        sparse.flat[indices] = target.flat[indices]
        mask.flat[indices] = 1.0
        return sparse, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        original_index, path = self.files[index]
        with np.load(path, allow_pickle=False) as sample:
            target = self._zero_one(sample["psi"])
            q = self._map(sample["q"], "q")
            height = float(np.asarray(sample["z"]).item())
            sample_id = str(np.asarray(sample["id"]).item()) if "id" in sample.files else path.stem

        if target.shape != q.shape:
            raise ValueError(f"Mismatched target/q shapes in {path}")
        building_id = sample_id.split("_", 1)[0]
        building_path = self.building_dir / f"{building_id}.json"
        if not building_path.is_file():
            raise FileNotFoundError(f"Missing building geometry: {building_path}")
        _, building_height = rasterize_buildings(
            str(building_path), target.shape[0], self.max_building_height
        )
        occupancy = (
            building_height * self.max_building_height >= height
        ).astype(np.float32)

        sparse, mask = self._sample(target, original_index)
        if np.any(sparse[mask == 0] != 0):
            raise RuntimeError("Sparse target leakage outside sampling mask")
        height_map = np.full_like(target, np.clip(height / self.max_height, 0.0, 1.0))
        return {
            "q": torch.from_numpy(q[None].astype(np.float32)),
            "building": torch.from_numpy(occupancy[None]),
            "building_height": torch.from_numpy(building_height[None].astype(np.float32)),
            "sparse": torch.from_numpy(sparse[None]),
            "mask": torch.from_numpy(mask[None]),
            "height_map": torch.from_numpy(height_map[None]),
            "height": torch.tensor(height, dtype=torch.float32),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "id": sample_id,
        }
