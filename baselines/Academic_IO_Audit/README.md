# Academic Input/Output Audit

This folder provides a model-independent audit of the input/output contract used
for the path-loss comparison. It does not train a model and does not alter any
dataset file.

## Publication contract

- Target: normalized path-loss map `psi`, shape `[1, 256, 256]` per altitude.
- Authorized observation: exactly 10% of `psi`, exposed only through `s` and
  `s_mask`.
- Unobserved target values must never occur in a model input.
- Shared raw geometry: transmitter map `q`, building information, and altitude.
- Proposed-only derived geometry: `depth` and `los`, deterministically generated
  from transmitter/building geometry. Their preprocessing must be published.
- Prediction: one path-loss map with the same shape and normalization as `psi`.
- Primary full-map protocol: apply the same final observation projection to all
  sampled methods, then calculate per-slice metrics and macro-average all test
  slices. Also retain unobserved-outdoor metrics as a secondary report.

## Run

```powershell
cd "D:\zdata\3DAFF-RadioDiff_Baselines\Academic_IO_Audit"
conda activate radiodiffzcy
python audit_io_contract.py --dataset-root "D:\datas\processed_pl_aff_dataset" --sample-rate 0.10
.\run_all_model_checks.ps1 -Gpu 0

# Or without activating Conda first:
.\run_all_model_checks.ps1 -Gpu 0 -PythonExe "D:\Anaconda3\envs\rmdm5070\python.exe"
```

The script writes `audit_report.json`. A failed invariant exits with a non-zero
status. Passing this audit verifies the data interface, not model accuracy.

`common_io.py` is the reusable reference implementation for sparse-input
construction, leakage assertions, output-shape checks, final measurement
projection, and per-slice error metrics. Import these functions into the final
evaluation scripts instead of rewriting the protocol separately for each model.

The current projects generate deterministic evaluation masks from each file's
position in the sorted split list. Therefore every method must use the same
sorted file list. Keep the split manifests with the released experiment.

For actual model-forward checks, run `run_all_model_checks.ps1`. The aligned
UniRM adapter in `D:\zdata\UniRM-baseline` uses only transmitter
position, raw building information, receiver altitude, and 10% sparse PL.
Earlier LoS/depth-conditioned UniRM checkpoints are not compatible and must not
be reported as results of the aligned adapter. The PEFNet-inspired adapter in
`C:\Users\Administrator\Documents\idea1\PEFNet-baseline` now follows the same
information budget and ignores NPZ `los/depth`; its older LoS-based checkpoints
are likewise incompatible and require retraining.
The 3D-RadioDiff check requires `RadioDiff3D\run_prepare.ps1` once after copying
the source-only bundle because generated manifests are intentionally excluded.
