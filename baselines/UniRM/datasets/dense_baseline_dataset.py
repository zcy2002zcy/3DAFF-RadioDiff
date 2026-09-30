"""Dense 256x256 inputs for image-to-image public baseline adaptations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from datasets.coordinate_query_dataset import MapRecord, load_raw_pathloss, read_manifest


def random_observation_mask(
    valid: np.ndarray, sample_rate: float, *, seed: int, map_index: int, epoch: int = 0
) -> np.ndarray:
    """Use exactly the same seeded random-mask protocol as GeoRadioTwin."""
    mask = np.zeros(valid.shape, dtype=np.bool_)
    if sample_rate <= 0:
        return mask
    eligible = np.flatnonzero(valid.ravel())
    count = max(1, int(round(len(eligible) * sample_rate)))
    rng = np.random.default_rng(seed + 17_171 + epoch * 1_000_003 + map_index * 97_409)
    mask.ravel()[rng.choice(eligible, count, replace=False)] = True
    return mask


class DenseBaselineDataset(Dataset[dict[str, Tensor]]):
    """Return building/source/height/sparse channels and a dense target."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_records: int,
        selection_strategy: str,
        seed: int,
        sample_rate: float,
        training: bool,
        exclude_target_values: Sequence[float] = (0,),
    ) -> None:
        self.records: list[MapRecord] = read_manifest(
            manifest_path,
            max_records=max_records,
            selection_strategy=selection_strategy,
            seed=seed,
        )
        self.seed = int(seed)
        self.sample_rate = float(sample_rate)
        self.training = bool(training)
        self.exclude_target_values = tuple(float(v) for v in exclude_target_values)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        target = load_raw_pathloss(record.pathloss_path)
        if record.building_height_path is None:
            raise ValueError("Dense baseline requires building heights")
        with Image.open(record.building_height_path) as image:
            building = np.asarray(image, dtype=np.float32)
        valid = building < record.rx_height
        for value in self.exclude_target_values:
            valid &= ~np.isclose(target, value, atol=0.0, rtol=0.0)
        epoch = self.epoch if self.training else 0
        observed = random_observation_mask(
            valid, self.sample_rate, seed=self.seed, map_index=index, epoch=epoch
        )
        height, width = target.shape
        source = np.zeros_like(target, dtype=np.float32)
        tx_row = int(round((height - 1) - record.tx_x))
        tx_col = int(round(record.tx_y))
        source[np.clip(tx_row, 0, height - 1), np.clip(tx_col, 0, width - 1)] = 1.0
        channels = [
            np.clip(building / 255.0, 0.0, 1.0),
            source,
            np.full_like(target, record.rx_height / 20.0, dtype=np.float32),
        ]
        if self.sample_rate > 0:
            channels.extend((np.where(observed, target / 255.0, 0.0), observed.astype(np.float32)))
        return {
            "input": torch.from_numpy(np.stack(channels).astype(np.float32)),
            "target": torch.from_numpy((target / 255.0).astype(np.float32)[None]),
            "target_raw": torch.from_numpy(target.astype(np.float32)[None]),
            "valid_mask": torch.from_numpy(valid[None]),
            "observation_mask": torch.from_numpy(observed[None]),
            "scene_id": torch.tensor(record.scene_id),
        }
