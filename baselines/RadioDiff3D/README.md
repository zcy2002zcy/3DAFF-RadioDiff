# RadioDiff-3D reimplementation: Path Loss / Random 10%

This is a transparent adaptation of the RadioDiff-3D method description for the same Path Loss dataset and evaluation contract used by 3DAFF-RadioDiff. It is not presented as the authors' unreleased official implementation.

## Condition input

The 3D condition tensor has four channels:

1. transmitter-position map `q`;
2. building-segmentation map rasterized from the scene JSON;
3. normalized building-height map rasterized from the same JSON;
4. a random 10% sparse Path Loss map.

The full target `psi`, LoS, AFF depth, and unobserved 90% target values never enter the condition network. Building vertices are transformed from Cartesian coordinates to image coordinates with `(x, y) -> (x-1, 256-y)`, which was validated against the building-valued pixels in the local radio maps. Test masks use each NPZ file's sorted split index as the seed, exactly matching the main 3DAFF-RadioDiff dataset implementation.

## Model and protocol

- Four altitude slices `z={1,2,3,4}` form one `[1,4,256,256]` volume.
- Residual 3D U-Net with 3D convolutions and bottleneck attention.
- The publication configuration uses four input frames, batch size 2, and learning rate `1e-4`.
- The publication-standard local configuration uses an MSE noise-prediction objective.
- EMA decay `0.995`, updated every 10 optimization steps.
- The project-compatible schedule uses 200 linear diffusion steps (`0.0005` to `0.10`) so the terminal state is effectively Gaussian, matching the primary 3DAFF-RadioDiff protocol.
- Evaluation starts from `x_T ~ N(0,I)` and uses deterministic 200-step DDIM.
- Final measurement projection uses the same random 10% mask. Pre- and post-projection metrics are both saved.
- NMSE, RMSE, SSIM, and PSNR are calculated per 2D altitude slice and then averaged, exactly matching the main project grain.

The public paper and dataset repository do not provide every implementation detail needed for bitwise reproduction. This project must therefore be reported as a reimplementation. `base_channels=32`, the bottleneck-attention implementation, EMA settings, and the project-compatible 200-step schedule are disclosed reproduction choices. The standard configuration selects `best.pt` using full-noise reconstruction NMSE on 20 fixed validation volumes; test data are not used for model selection.

## Publication-standard commands

Run these commands for the result used in the comparison table. They write to
`outputs/radiodiff3d_pl_random10_standard` and do not overwrite the earlier run.

```powershell
cd "C:\Users\Administrator\Desktop\3d-radiodiff - 副本\3DAFF-RadioDiff\baselines\radiodiff3d"

# Dataset manifests are shared with the earlier run; prepare only if absent.
.\run_prepare.ps1 -Workers 4
.\run_audit.ps1
.\run_check_standard.ps1 -Gpu 0

# Train from scratch.
.\run_train_standard.ps1 -Gpu 0

# Resume only after an interruption.
.\run_train_standard.ps1 -Gpu 0 -Resume

# Smoke evaluation: 100 volumes / 400 slices.
.\run_eval_standard.ps1 -Gpu 0 -MaxVolumes 100

# Final evaluation: 10,000 volumes / 40,000 slices.
.\run_eval_standard.ps1 -Gpu 0 -MaxVolumes 0
```

The primary table uses post-projection metrics, because every sampled method is
given the same measured pixels. The JSON output also records all four metrics
before projection. Apply the same convention and metric implementation to every
sampled baseline and to 3DAFF-RadioDiff.

## Commands

```powershell
cd "C:\Users\Administrator\Desktop\3d-radiodiff - 副本\3DAFF-RadioDiff\baselines\radiodiff3d"

# Recover complete volumes from train+val, split them by scene ID, and index test
.\run_prepare.ps1 -Workers 4

# Mandatory before publication training
.\run_audit.ps1

# Verify geometry, conditions, model shape, and terminal noise
.\run_check.ps1

# Earlier disclosed approximation (kept only for traceability)
.\run_train.ps1 -Gpu 0

# Quick evaluation: 100 volumes = 400 slices
.\run_eval.ps1 -Gpu 0 -MaxVolumes 100

# Full evaluation: 10,000 volumes = 40,000 slices
.\run_eval.ps1 -Gpu 0 -MaxVolumes 0
```

The source train/validation folders were originally split at the 2D-slice level. `run_prepare.ps1` therefore combines only those two development folders and creates a deterministic 90/10 scene-ID split with complete four-altitude volumes. The test folder is never merged or changed. For a strictly paired publication comparison, 3DAFF-RadioDiff should be retrained using the same corrected scene-level split.

For the current local dataset, preparation produced 108,076 training volumes, 12,008 validation volumes, and 10,000 unchanged test volumes. It excluded 116 incomplete development scenes that lacked at least one of the four required altitude slices; no values or missing slices were imputed.
