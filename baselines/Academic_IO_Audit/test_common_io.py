from __future__ import annotations

import torch

from common_io import (
    apply_final_measurement_projection,
    assert_sparse_is_leakage_safe,
    full_slice_error_metrics,
    make_sparse_observation,
)


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    target = torch.rand((2, 1, 32, 32), generator=generator) * 2.0 - 1.0
    mask = torch.zeros_like(target)
    mask.flatten(1)[:, :102] = 1.0
    sparse = make_sparse_observation(target, mask)
    assert_sparse_is_leakage_safe(sparse, target, mask)
    prediction = torch.zeros_like(target)
    projected = apply_final_measurement_projection(prediction, sparse, mask)
    assert torch.equal(projected[mask.bool()], target[mask.bool()])
    metrics = full_slice_error_metrics(projected, target)
    assert all(value.shape == (2,) for value in metrics.values())
    print("COMMON I/O SELF-TEST PASSED")


if __name__ == "__main__":
    main()
