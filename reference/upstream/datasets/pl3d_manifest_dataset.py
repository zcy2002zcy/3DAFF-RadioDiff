"""Leakage-safe h1--h4 volume loader for the raw path-loss manifests.

The loader keeps the raw PNG files immutable and builds one 3-D sample from
four receiver heights lazily.  Sparse measurements are sampled only from
physically valid receiver voxels; the measurement mask is returned separately
so an unobserved voxel can never be confused with a measured zero value.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.idw_prior import make_idw_prior


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != (256, 256):
        raise ValueError(f"Expected a 256x256 path-loss map, got {array.shape}: {path}")
    return array.astype(np.uint8, copy=False)


class PL3DManifestDataset(Dataset[dict[str, Any]]):
    """Return one h1--h4 volume per scene/transmitter pair.

    Tensor shapes use ``[C,D,H,W]``.  ``target`` and ``measurement`` use the
    model range [-1, 1].  Geometry is derived only from building metadata and
    coordinates, never from target path-loss pixels.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        sample_rate: float = 0.10,
        sampling_mode: str = "random",
        deterministic_sampling: bool | None = None,
        seed: int = 42,
        condition_profile: str = "full",
        include_idw_prior: bool = False,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        sparse_profiles = {"q_building_sparse", "q_building_random10"}
        if condition_profile not in {
            "full",
            "aff_prediction",
            "q_building",
            *sparse_profiles,
        }:
            raise ValueError(
                "condition_profile must be full, aff_prediction, q_building, "
                "or q_building_sparse"
            )
        if condition_profile in {"full", *sparse_profiles} and not 0.0 < sample_rate <= 1.0:
            raise ValueError("sampling profiles require sample_rate in (0, 1]")
        if condition_profile in {"q_building", "aff_prediction"} and not 0.0 <= sample_rate <= 1.0:
            raise ValueError("prediction profiles require sample_rate in [0, 1]")
        if sampling_mode not in {"random", "uniform"}:
            raise ValueError("sampling_mode must be random or uniform")

        self.sample_rate = float(sample_rate)
        self.condition_profile = condition_profile
        self.sampling_mode = sampling_mode
        self.deterministic_sampling = (
            self.split != "train"
            if deterministic_sampling is None
            else bool(deterministic_sampling)
        )
        self.seed = int(seed)
        self.include_idw_prior = bool(include_idw_prior)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        self.packed_npz = bool(self.metadata.get("packed_npz", False))
        self.raw_root = Path(self.metadata["raw_root"])
        self.los_root = Path(self.metadata["los_root"])
        self.depth_root = Path(self.metadata["first_obstruction_depth_root"])
        self.heights = tuple(float(v) for v in self.metadata["receiver_heights_m"])
        if self.heights != (1.0, 2.0, 3.0, 4.0):
            raise ValueError(f"This experiment requires receiver heights 1--4 m, got {self.heights}")
        self.tx_height = float(self.metadata["transmitter_height_m"])
        if not np.isclose(self.tx_height, 1.5):
            raise ValueError(f"Expected fixed transmitter height 1.5 m, got {self.tx_height}")

        with (self.root / f"{self.split}.csv").open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty manifest split: {self.split}")

        row_grid, col_grid = np.meshgrid(
            np.arange(256, dtype=np.float32),
            np.arange(256, dtype=np.float32),
            indexing="ij",
        )
        self._row_grid = row_grid
        self._col_grid = col_grid
        self._distance_scale = float(np.sqrt(255.0**2 + 255.0**2 + 2.5**2))

    def __len__(self) -> int:
        return len(self.rows)

    def _png_path(self, row: dict[str, str], height: int) -> Path:
        suffix = "train" if row["raw_partition"] == "train" else "test1"
        return (
            self.raw_root
            / f"PathLoss_h{height}_{suffix}"
            / f"h{height}"
            / f"{row['sample_id']}.png"
        )

    @staticmethod
    def _geometry_path(root: Path, row: dict[str, str], height: int) -> Path:
        suffix = "train" if row["raw_partition"] == "train" else "test1"
        return (
            root
            / f"PathLoss_h{height}_{suffix}"
            / f"h{height}"
            / f"{row['sample_id']}.npy"
        )

    def _load_geometry_volume(self, root: Path, row: dict[str, str]) -> np.ndarray:
        arrays = []
        for height in (int(value) for value in self.heights):
            path = self._geometry_path(root, row, height)
            array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            if array.shape != (256, 256):
                raise ValueError(f"Expected 256x256 geometry map, got {array.shape}: {path}")
            arrays.append(array)
        return np.stack(arrays, axis=0)

    def _sample_mask(self, valid: np.ndarray, index: int) -> np.ndarray:
        flat_valid = np.flatnonzero(valid.reshape(-1))
        count = max(1, int(round(len(flat_valid) * self.sample_rate)))
        if self.sampling_mode == "uniform":
            chosen = flat_valid[np.linspace(0, len(flat_valid) - 1, count, dtype=np.int64)]
        else:
            rng = np.random.default_rng(self.seed + index if self.deterministic_sampling else None)
            chosen = rng.choice(flat_valid, size=count, replace=False)
        mask = np.zeros(valid.size, dtype=np.float32)
        mask[chosen] = 1.0
        return mask.reshape(valid.shape)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        height_ints = tuple(int(v) for v in self.heights)
        if self.packed_npz:
            packed_path = (
                self.root
                / "packed"
                / self.split
                / f"scene_{int(row['scene_id']):04d}"
                / f"{row['sample_id']}.npz"
            )
            with np.load(packed_path, allow_pickle=False) as packed:
                target_u8 = packed["target_u8"].astype(np.uint8, copy=False)
                if self.condition_profile in {"full", "aff_prediction"}:
                    los = packed["los_u8"].astype(np.float32)
                    first_obstruction_depth = packed["depth_f16"].astype(np.float32)
        else:
            target_u8 = np.stack(
                [_load_gray(self._png_path(row, height)) for height in height_ints], axis=0
            )
            if self.condition_profile in {"full", "aff_prediction"}:
                los = self._load_geometry_volume(self.los_root, row)
                first_obstruction_depth = self._load_geometry_volume(self.depth_root, row)
        target = target_u8.astype(np.float32) / 127.5 - 1.0
        if self.condition_profile in {"full", "aff_prediction"}:
            if not np.isin(los, (0.0, 1.0)).all():
                raise ValueError(f"LoS map is not binary for sample {row['sample_id']}")
            if first_obstruction_depth.min() < 0.0 or first_obstruction_depth.max() > 1.0:
                raise ValueError(
                    f"First-obstruction depth is outside [0,1] for sample {row['sample_id']}"
                )
        else:
            # Keep the public tensor schema stable without reading privileged
            # or unused geometry from disk.
            los = np.zeros_like(target, dtype=np.float32)
            first_obstruction_depth = np.zeros_like(target, dtype=np.float32)

        scene_id = int(row["scene_id"])
        with np.load(self.root / "geometry" / f"scene_{scene_id:04d}.npz") as data:
            building_height = data["building_height_m"].astype(np.float32)
        receiver_heights = np.asarray(self.heights, dtype=np.float32)[:, None, None]
        valid = building_height[None] < receiver_heights

        tx_row, tx_col = int(row["tx_row"]), int(row["tx_col"])
        q = np.zeros_like(target, dtype=np.float32)
        q[:, tx_row, tx_col] = 1.0
        if self.condition_profile == "full":
            dz = receiver_heights - self.tx_height
            distance = np.sqrt(
                (self._row_grid[None] - tx_row) ** 2
                + (self._col_grid[None] - tx_col) ** 2
                + dz**2
            ).astype(np.float32) / self._distance_scale
            measurement_mask = self._sample_mask(valid, index)
            measurement = np.where(measurement_mask > 0, target, 0.0).astype(np.float32)
            height_volume = np.broadcast_to(receiver_heights / 4.0, target.shape)
        elif self.condition_profile in {"q_building_sparse", "q_building_random10"}:
            distance = np.zeros_like(target, dtype=np.float32)
            measurement_mask = self._sample_mask(valid, index)
            measurement = np.where(measurement_mask > 0, target, 0.0).astype(np.float32)
            height_volume = np.zeros_like(target, dtype=np.float32)
        else:
            distance = np.zeros_like(target, dtype=np.float32)
            measurement_mask = np.zeros_like(target, dtype=np.float32)
            measurement = np.zeros_like(target, dtype=np.float32)
            height_volume = np.zeros_like(target, dtype=np.float32)

        depth = len(self.heights)
        building_volume = np.broadcast_to(building_height[None], target.shape)
        condition = np.stack(
            (
                np.broadcast_to((building_volume > 0).astype(np.float32), target.shape),
                np.clip(building_volume / 20.0, 0.0, 1.0),
                q,
                height_volume,
                distance,
                los,
                first_obstruction_depth,
                measurement,
                measurement_mask,
            ),
            axis=0,
        ).astype(np.float32)
        assert condition.shape == (9, depth, 256, 256)

        idw_prior = (
            make_idw_prior(measurement, measurement_mask)
            if self.include_idw_prior
            else np.zeros_like(target, dtype=np.float32)
        )

        return {
            "id": row["sample_id"],
            "condition": torch.from_numpy(condition.copy()),
            "target": torch.from_numpy(target[None].copy()),
            "target_u8": torch.from_numpy(target_u8[None].copy()),
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "los": torch.from_numpy(los[None].copy()),
            "first_obstruction_depth": torch.from_numpy(
                first_obstruction_depth[None].copy()
            ),
            "measurement": torch.from_numpy(measurement[None].copy()),
            "measurement_mask": torch.from_numpy(measurement_mask[None].copy()),
            "idw_prior": torch.from_numpy(idw_prior[None].copy()),
            "scene_id": torch.tensor(scene_id, dtype=torch.long),
            "tx_position": torch.tensor(
                [float(row["tx_x_token"]), float(row["tx_y_token"]), self.tx_height],
                dtype=torch.float32,
            ),
            "rx_heights": torch.tensor(self.heights, dtype=torch.float32),
        }
