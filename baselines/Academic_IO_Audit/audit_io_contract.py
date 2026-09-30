"""Audit the publication input/output contract without modifying source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_CONTRACTS: dict[str, dict[str, Any]] = {
    "3DAFF-RadioDiff": {
        "inputs": ["q", "depth", "los", "s", "s_mask", "z", "x_t", "t"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "publication_candidate",
    },
    "U-Net": {
        "inputs": ["q", "building_mask", "s", "s_mask", "z"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "publication_candidate_after_matched_evaluation",
    },
    "ResNet": {
        "inputs": ["q", "building_mask", "s", "s_mask", "z"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "publication_candidate_after_matched_evaluation",
    },
    "Pix2Pix": {
        "inputs": ["q", "building_mask", "s", "s_mask", "z"],
        "target": "psi (generator loss/discriminator real target/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "publication_candidate_after_matched_evaluation",
    },
    "IDW": {
        "inputs": ["observed coordinates", "observed values", "s_mask"],
        "target": "psi (metrics only)",
        "output": "interpolated path-loss slice [1,H,W]",
        "status": "publication_candidate_after_matched_evaluation",
    },
    "Kriging": {
        "inputs": ["observed coordinates", "observed values", "s_mask"],
        "target": "psi (metrics only)",
        "output": "kriged path-loss slice [1,H,W]",
        "status": "publication_candidate_after_full_test_evaluation",
    },
    "Kriging+topology": {
        "inputs": ["observed coordinates", "observed values", "s_mask", "building_mask"],
        "target": "psi (metrics only)",
        "output": "topology-aware kriged path-loss slice [1,H,W]",
        "status": "publication_candidate_after_full_test_evaluation",
    },
    "3D-RadioDiff (reimplementation)": {
        "inputs": ["q", "building_mask", "building_height", "s", "s_mask", "z", "x_t", "t"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss volume [1,Z,H,W]",
        "status": "publication_candidate_with_reimplementation_label",
    },
    "PEFNet-inspired": {
        "inputs": ["q", "receiver-height building occupancy", "s", "s_mask", "z"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "retraining_required_after_input_alignment",
    },
    "UniRM building-input adapter": {
        "inputs": ["q", "building occupancy/height", "s", "s_mask", "z"],
        "target": "psi (loss/metrics only)",
        "output": "predicted path-loss slice [1,H,W]",
        "status": "retraining_required_after_input_alignment",
    },
}


def make_shared_mask(shape: tuple[int, int], rate: float, seed: int) -> np.ndarray:
    count = int(round(np.prod(shape) * rate))
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.prod(shape), size=count, replace=False)
    mask = np.zeros(np.prod(shape), dtype=np.float32)
    mask[indices] = 1.0
    return mask.reshape(shape)


def scalar(value: np.ndarray) -> Any:
    item = np.asarray(value).reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def audit_sample(path: Path, rate: float, sorted_split_index: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        required = {"q", "psi", "z", "id"}
        missing = sorted(required - keys)
        if missing:
            raise AssertionError(f"{path.name}: missing required keys {missing}")
        q = np.asarray(archive["q"], dtype=np.float32)
        psi = np.asarray(archive["psi"], dtype=np.float32)
        z = float(scalar(archive["z"]))
        sample_id = str(scalar(archive["id"]))
        derived = {name: np.asarray(archive[name], dtype=np.float32) for name in ("depth", "los") if name in keys}

    if q.shape != psi.shape or q.ndim != 2:
        raise AssertionError(f"{path.name}: q/psi must be matching 2D arrays, got {q.shape}/{psi.shape}")
    if not np.isfinite(q).all() or not np.isfinite(psi).all():
        raise AssertionError(f"{path.name}: q or psi contains NaN/Inf")
    for name, value in derived.items():
        if value.shape != psi.shape or not np.isfinite(value).all():
            raise AssertionError(f"{path.name}: invalid {name} shape or values")

    # Matches the current 3DAFF and aligned-baseline evaluation loaders.
    mask = make_shared_mask(psi.shape, rate, sorted_split_index)
    sparse = psi * mask
    expected_count = int(round(psi.size * rate))
    if int(mask.sum()) != expected_count:
        raise AssertionError(f"{path.name}: mask count mismatch")
    if np.count_nonzero(sparse[mask == 0]) != 0:
        raise AssertionError(f"{path.name}: unobserved target leaked into sparse input")
    if not np.array_equal(sparse[mask == 1], psi[mask == 1]):
        raise AssertionError(f"{path.name}: observed values do not match target")

    # A deliberately corrupted unobserved target must not change any authorized input.
    corrupted = psi.copy()
    corrupted[mask == 0] += 123.0
    sparse_corrupted = corrupted * mask
    if not np.array_equal(sparse, sparse_corrupted):
        raise AssertionError(f"{path.name}: authorized input depends on unobserved target")

    return {
        "file": str(path),
        "id": sample_id,
        "altitude": z,
        "keys": sorted(keys),
        "shape": list(psi.shape),
        "sample_count": expected_count,
        "sample_fraction": float(mask.mean()),
        "sorted_split_index_seed": sorted_split_index,
        "q_range": [float(q.min()), float(q.max())],
        "target_range": [float(psi.min()), float(psi.max())],
        "derived_geometry_present": sorted(derived),
        "unobserved_target_invariance": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--output", default="audit_report.json")
    args = parser.parse_args()

    if not 0.0 < args.sample_rate < 1.0:
        raise ValueError("sample-rate must be between 0 and 1")
    root = Path(args.dataset_root)
    audited: dict[str, list[dict[str, Any]]] = {}
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        files = sorted(split_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No NPZ files in {split_dir}")
        selected = list(enumerate(files[: max(1, args.samples_per_split)]))
        audited[split] = [
            audit_sample(path, args.sample_rate, original_index)
            for original_index, path in selected
        ]
        split_ids[split] = {item["id"] for item in audited[split]}

    overlaps = {
        "train_val": sorted(split_ids["train"] & split_ids["val"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "val_test": sorted(split_ids["val"] & split_ids["test"]),
    }
    # This is a smoke check over selected records. Full split auditing should use manifests.
    report = {
        "status": "PASS",
        "dataset_root": str(root.resolve()),
        "sample_rate": args.sample_rate,
        "mask_definition": "sorted split index as NumPy RNG seed, exact without-replacement sample count",
        "metric_contract": {
            "primary": "full slice after final measurement projection",
            "secondary": "unobserved non-building pixels",
            "aggregation": "per-slice metrics followed by macro average",
            "normalization": "shared target normalization and PSNR data_range",
        },
        "sample_audits": audited,
        "selected_record_id_overlaps": overlaps,
        "method_contracts": METHOD_CONTRACTS,
        "warnings": [
            "Depth/LoS preprocessing provenance must be audited separately and published.",
            "The aligned PEFNet-inspired adapter uses q, raw building information, sparse PL, and altitude; old LoS-based checkpoints are incompatible.",
            "The aligned UniRM adapter uses q, raw building information, sparse PL, and altitude; old checkpoints are incompatible.",
            "All methods must consume the same saved masks keyed by sample ID and altitude in the final experiment.",
        ],
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"AUDIT PASSED: {output.resolve()}")


if __name__ == "__main__":
    main()
