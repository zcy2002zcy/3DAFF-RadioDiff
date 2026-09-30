# Strict Kriging baselines

- `kriging`: fixed random-10% sparse measurements only.
- `kriging_with_topology`: the same measurements plus the raw building mask.
- PyKrige is mandatory. There is no RBF/IDW fallback hidden under the Kriging name.
- Both methods use local 32-neighbor Ordinary Kriging, output clipping, final
  measurement projection, and the shared four-metric implementation.

```powershell
.\run_check_kriging.ps1 -Gpu 2

# Run the two methods separately
.\run_eval_kriging_only.ps1 -Gpu 2 -MaxSamples 0
.\run_eval_kriging_with_topology.ps1 -Gpu 2 -MaxSamples 0

# Or run both methods together; use MaxSamples 1 for a smoke test
.\run_eval_kriging.ps1 -Gpu 2 -MaxSamples 1

# Full 40,000-image publication evaluation (computationally expensive)
.\run_eval_kriging.ps1 -Gpu 2 -MaxSamples 0
```

If a reduced deterministic subset is used, every compared method must be evaluated on
the identical subset and the sample count/seed must be stated in the paper.
