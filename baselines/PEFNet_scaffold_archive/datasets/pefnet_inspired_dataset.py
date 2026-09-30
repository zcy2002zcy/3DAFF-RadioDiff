"""Leakage-safe dense inputs for the PEFNet-inspired adapted baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from datasets.coordinate_query_dataset import MapRecord, load_raw_pathloss, read_manifest
from geometry.coordinates import physical_xy_to_row_col
from physics.vie_operator import build_incident_field, building_contrast


PEFNET_INPUT_CHANNELS = (
    "tx_source",
    "building_occupancy_at_rx_height",
    "incident_real",
    "incident_imag",
    "rx_height",
)


class PEFNetInspiredDataset(Dataset[dict[str, Tensor]]):
    """Return map inputs without exposing target values in any input channel.

    This is a documented 2.5-D adaptation.  The first four channels reproduce
    the feature families described by PEFNet, while a fifth receiver-height
    plane is required to distinguish UrbanRadio3D h1--h4 maps with one model.
    The target is used only as supervision and to apply the project's existing
    invalid-value evaluation mask.
    """

    input_channel_names = PEFNET_INPUT_CHANNELS

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_records: int | None,
        selection_strategy: str,
        seed: int,
        sample_rate: float = 0.0,
        training: bool = False,
        dimensionless_wavenumber: float = 0.35,
        grid_spacing: float = 1.0,
        epsilon_air: float = 1.0,
        epsilon_building: float = 4.0,
        conductivity: float = 0.0,
        rx_height_scale: float = 20.0,
        target_scale: float = 255.0,
        pathloss_channel: int = 0,
        exclude_building_occupied: bool = True,
        exclude_target_values: Sequence[float] = (0,),
        target_exclusion_tolerance: float = 0.0,
    ) -> None:
        if sample_rate != 0.0:
            raise ValueError("PEFNet-inspired baseline is sampling-free; sample_rate must be 0")
        if rx_height_scale <= 0 or target_scale <= 0:
            raise ValueError("rx_height_scale and target_scale must be positive")
        if target_exclusion_tolerance < 0:
            raise ValueError("target_exclusion_tolerance must be non-negative")
        self.records: list[MapRecord] = read_manifest(
            manifest_path,
            max_records=max_records,
            selection_strategy=selection_strategy,
            seed=seed,
        )
        self.seed = int(seed)
        self.training = bool(training)
        self.dimensionless_wavenumber = float(dimensionless_wavenumber)
        self.grid_spacing = float(grid_spacing)
        self.epsilon_air = float(epsilon_air)
        self.epsilon_building = float(epsilon_building)
        self.conductivity = float(conductivity)
        self.rx_height_scale = float(rx_height_scale)
        self.target_scale = float(target_scale)
        self.pathloss_channel = int(pathloss_channel)
        self.exclude_building_occupied = bool(exclude_building_occupied)
        self.exclude_target_values = tuple(float(value) for value in exclude_target_values)
        self.target_exclusion_tolerance = float(target_exclusion_tolerance)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # Kept for parity with the other training datasets.  Inputs are fully
        # deterministic and do not resample labels or observations.
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        if record.building_height_path is None:
            raise ValueError("PEFNet-inspired baseline requires a building raster")
        target = load_raw_pathloss(record.pathloss_path, self.pathloss_channel)
        with Image.open(record.building_height_path) as image:
            building_height = np.asarray(image, dtype=np.float32)
        if building_height.shape != target.shape:
            raise ValueError(
                f"Building/PathLoss shape mismatch: {building_height.shape} vs {target.shape}"
            )
        height, width = target.shape
        tx_row_float, tx_col_float = physical_xy_to_row_col(
            record.tx_x, record.tx_y, height
        )
        tx_row = int(round(tx_row_float))
        tx_col = int(round(tx_col_float))
        if not (0 <= tx_row < height and 0 <= tx_col < width):
            raise ValueError(
                f"Tx raster coordinate {(tx_row, tx_col)} outside {(height, width)}"
            )

        source = np.zeros((height, width), dtype=np.float32)
        source[tx_row, tx_col] = 1.0
        # A horizontal 2.5-D slice is the most conservative way to adapt the
        # paper's binary 2-D layout to height-indexed UrbanRadio3D maps.
        occupancy = building_height >= record.rx_height
        incident = build_incident_field(
            (height, width),
            (tx_row, tx_col),
            dimensionless_wavenumber=self.dimensionless_wavenumber,
            grid_spacing=self.grid_spacing,
            normalize=True,
        )
        height_plane = np.full(
            (height, width),
            record.rx_height / self.rx_height_scale,
            dtype=np.float32,
        )
        inputs = np.stack(
            (
                source,
                occupancy.astype(np.float32),
                incident[0],
                incident[1],
                height_plane,
            ),
            axis=0,
        ).astype(np.float32, copy=False)

        chi = building_contrast(
            occupancy,
            epsilon_air=self.epsilon_air,
            epsilon_building=self.epsilon_building,
            conductivity=self.conductivity,
        )
        contrast = np.stack((chi.real, chi.imag), axis=0).astype(np.float32, copy=False)

        valid = np.ones(target.shape, dtype=np.bool_)
        if self.exclude_building_occupied:
            valid &= building_height < record.rx_height
        for excluded in self.exclude_target_values:
            valid &= ~np.isclose(
                target,
                excluded,
                atol=self.target_exclusion_tolerance,
                rtol=0.0,
            )

        # A zero observation mask preserves the common no-sampling evaluator
        # contract.  Crucially, no target-derived value enters ``input``.
        observation_mask = np.zeros((1, height, width), dtype=np.bool_)
        return {
            "input": torch.from_numpy(inputs),
            "target": torch.from_numpy(
                (target / self.target_scale).astype(np.float32, copy=False)[None]
            ),
            "target_raw": torch.from_numpy(target.astype(np.float32, copy=False)[None]),
            "valid_mask": torch.from_numpy(valid[None]),
            "observation_mask": torch.from_numpy(observation_mask),
            "incident_field": torch.from_numpy(incident),
            "contrast": torch.from_numpy(contrast),
            "building_mask": torch.from_numpy(occupancy.astype(np.bool_)[None]),
            "scene_id": torch.tensor(record.scene_id, dtype=torch.int64),
            "map_index": torch.tensor(index, dtype=torch.int64),
            "rx_height": torch.tensor(record.rx_height, dtype=torch.float32),
            "tx_row_col": torch.tensor((tx_row, tx_col), dtype=torch.int64),
        }


__all__ = ["PEFNET_INPUT_CHANNELS", "PEFNetInspiredDataset"]
