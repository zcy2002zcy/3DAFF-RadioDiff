import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_sampling_map_from_psi(
    psi: np.ndarray,
    sample_rate: float = 0.10,
    sampling_mode: str = "random",
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if psi.ndim != 2:
        raise ValueError(f"psi must be [H,W], got {psi.shape}")
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate must be in (0,1], got {sample_rate}")

    height, width = psi.shape
    total = height * width
    count = max(1, int(round(total * sample_rate)))
    mode = sampling_mode.lower()
    if mode == "random":
        if seed is None:
            indices = np.random.choice(total, size=count, replace=False)
        else:
            indices = np.random.default_rng(seed).choice(total, size=count, replace=False)
    elif mode == "uniform":
        rows_n = max(1, int(round(np.sqrt(count * height / width))))
        cols_n = max(1, int(round(count / rows_n)))
        rows = np.linspace(0, height - 1, rows_n, dtype=np.int64)
        cols = np.linspace(0, width - 1, cols_n, dtype=np.int64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        indices = np.unique(rr.reshape(-1) * width + cc.reshape(-1))
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    sparse = np.zeros_like(psi, dtype=np.float32)
    mask = np.zeros_like(psi, dtype=np.float32)
    sparse.reshape(-1)[indices] = psi.reshape(-1)[indices]
    mask.reshape(-1)[indices] = 1.0
    return sparse, mask


@lru_cache(maxsize=128)
def rasterize_buildings(
    json_path: str,
    image_size: int,
    max_building_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    objects = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mask_image = Image.new("F", (image_size, image_size), 0.0)
    height_image = Image.new("F", (image_size, image_size), 0.0)
    mask_draw = ImageDraw.Draw(mask_image)
    height_draw = ImageDraw.Draw(height_image)
    for vertices, height_container in objects:
        value = float(np.asarray(height_container).reshape(-1)[0])
        polygon = [
            (float(x) - 1.0, float(image_size) - float(y))
            for x, y in vertices
        ]
        mask_draw.polygon(polygon, fill=1.0)
        normalized = min(max(value / max_building_height, 0.0), 1.0)
        height_draw.polygon(polygon, fill=normalized)
    return (
        np.asarray(mask_image, dtype=np.float32),
        np.asarray(height_image, dtype=np.float32),
    )


class RadioMapBuildingDataset(Dataset):
    """PL slices with raw BS/building conditions and sparse observations only."""

    def __init__(
        self,
        data_dir: str,
        building_dir: str,
        image_size: int = 256,
        max_building_height: float = 20.0,
        use_sampling: bool = True,
        sample_rate: float = 0.10,
        sampling_mode: str = "random",
        deterministic_sampling: bool = False,
        verify_files: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.building_dir = Path(building_dir)
        self.image_size = int(image_size)
        self.max_building_height = float(max_building_height)
        self.use_sampling = bool(use_sampling)
        self.sample_rate = float(sample_rate)
        self.sampling_mode = sampling_mode.lower()
        self.deterministic_sampling = bool(deterministic_sampling)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        if not self.building_dir.exists():
            raise FileNotFoundError(f"building_dir not found: {self.building_dir}")
        self.files: List[Path] = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No NPZ files found in {self.data_dir}")
        print(f"[Dataset] {self.data_dir} -> {len(self.files)} samples")
        if verify_files:
            self._verify_files()

    def _verify_files(self) -> None:
        for path in self.files:
            with np.load(path, allow_pickle=True) as data:
                for key in ("q", "psi", "id", "z"):
                    if key not in data.files:
                        raise KeyError(f"{path}: missing key {key}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=True) as data:
            q = data["q"].astype(np.float32)
            psi = data["psi"].astype(np.float32)
            sample_id = str(np.asarray(data["id"]).item())
            altitude = np.float32(np.asarray(data["z"]).item())

        expected = (self.image_size, self.image_size)
        if q.shape != expected or psi.shape != expected:
            raise ValueError(f"{path}: q={q.shape}, psi={psi.shape}, expected={expected}")

        building_id = sample_id.split("_", 1)[0]
        geometry_path = self.building_dir / f"{building_id}.json"
        if not geometry_path.exists():
            raise FileNotFoundError(f"Building geometry not found: {geometry_path}")
        building_mask, building_height = rasterize_buildings(
            str(geometry_path), self.image_size, self.max_building_height
        )

        sample: Dict[str, Any] = {
            "id": sample_id,
            "q": torch.from_numpy(q).unsqueeze(0),
            "building_mask": torch.from_numpy(building_mask.copy()).unsqueeze(0),
            "building_height": torch.from_numpy(building_height.copy()).unsqueeze(0),
            "psi": torch.from_numpy(psi).unsqueeze(0),
            "z": torch.tensor([altitude], dtype=torch.float32),
        }
        if self.use_sampling:
            seed = index if self.deterministic_sampling else None
            sparse, sparse_mask = make_sampling_map_from_psi(
                psi, self.sample_rate, self.sampling_mode, seed
            )
            if np.any(sparse[sparse_mask == 0] != 0):
                raise RuntimeError("Sparse target leakage outside sampling mask")
            sample["s"] = torch.from_numpy(sparse).unsqueeze(0)
            sample["s_mask"] = torch.from_numpy(sparse_mask).unsqueeze(0)
        return sample


def build_dataset_from_config(
    data_dir: str,
    cfg: Dict[str, Any],
    split: Optional[str] = None,
) -> Dataset:
    data_cfg = cfg.get("data", cfg)
    deterministic = bool(
        split in {"val", "test"}
        and data_cfg.get("deterministic_sampling_for_eval", True)
    )
    dataset: Dataset = RadioMapBuildingDataset(
        data_dir=data_dir,
        building_dir=str(data_cfg["building_dir"]),
        image_size=int(data_cfg.get("image_size", 256)),
        max_building_height=float(data_cfg.get("max_building_height", 20.0)),
        use_sampling=bool(data_cfg.get("use_sampling", True)),
        sample_rate=float(data_cfg.get("sample_rate", 0.10)),
        sampling_mode=str(data_cfg.get("sampling_mode", "random")),
        deterministic_sampling=deterministic,
        verify_files=bool(data_cfg.get("verify_files", False)),
    )
    subset_key = None
    if split == "train":
        subset_key = "random_train_samples"
    elif split == "val":
        subset_key = "random_val_samples"
    subset_count = int(data_cfg.get(subset_key, 0) or 0) if subset_key else 0
    if subset_count > 0:
        subset_count = min(subset_count, len(dataset))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        indices = rng.choice(len(dataset), size=subset_count, replace=False).tolist()
        dataset = Subset(dataset, indices)
    return dataset


def build_dataloader_from_config(
    data_dir: str,
    cfg: Dict[str, Any],
    shuffle: bool = False,
    drop_last: bool = False,
    split: Optional[str] = None,
) -> DataLoader:
    dataset = build_dataset_from_config(data_dir, cfg, split)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    batch_size = int(eval_cfg.get("batch_size", 1)) if split in {"val", "test"} else int(train_cfg.get("batch_size", 1))
    workers = int(data_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)) and workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )
