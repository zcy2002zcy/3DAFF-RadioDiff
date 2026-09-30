# PEFNet-inspired baseline: aligned PL random10 protocol

This folder is an independent, auditable **PEFNet-inspired adaptation**, not
the authors' official implementation. It is aligned with the information and
evaluation protocol of `3DAFF-RadioDiff-PathLoss`.

## Authorized information

- transmitter-position map `q`;
- raw Urban3D building polygons and heights;
- receiver altitude `z`;
- exactly 10% sparse path-loss observations and their binary mask.

The NPZ fields `los` and `depth` are deliberately ignored. Full `psi` is used
only as the training target, to fill locations selected by the sparse mask,
and for final metrics. The building occupancy at receiver height is used as a
documented susceptibility proxy because material permittivity is unavailable.

## Fixed comparison protocol

- data: `D:\datas\processed_pl_aff_dataset`;
- buildings: `D:\zdata\Building_Infomation\buildings`;
- random10: 6,554 of 65,536 pixels;
- validation/test masks: deterministic NumPy RNG seeded by sorted file index;
- test: all 40,000 held-out slices;
- final observation projection at measured pixels;
- per-slice macro averages of NMSE, RMSE, SSIM, and PSNR.

Old `pefnet_los_*` checkpoints are incompatible and must not be reported as
results of this aligned baseline.

## Run

```powershell
cd "C:\Users\Administrator\Documents\idea1\PEFNet-baseline"
conda activate rmdm5070
```

Check the information contract and a real forward pass:

```powershell
.\run_check_building_random10.ps1
```

Stage 1, physics/coarse branch:

```powershell
.\run_train_building_random10.ps1 -Stage physics -Gpu 0
```

Stage 2, supervised residual refiner:

```powershell
.\run_train_building_random10.ps1 -Stage refine -Gpu 0
```

Full 40,000-slice evaluation:

```powershell
.\run_eval_building_random10.ps1 -Gpu 0
```

Report the method as **PEFNet-inspired (building-input adaptation, our
implementation)**. Do not label it as official PEFNet.
