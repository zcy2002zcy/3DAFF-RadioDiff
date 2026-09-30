# 3DAFF-RadioDiff Path-Loss Baselines

This folder consolidates the baseline code associated with the path-loss
comparison table. It contains source code and configurations only. Datasets,
checkpoints, logs, caches, and generated evaluation outputs are intentionally
excluded.

## Included methods

| Folder | Method | Status |
| --- | --- | --- |
| `Kriging` | Ordinary Kriging and Kriging + topology | Runnable local implementation; both variants share one folder. |
| `RadioDiff3D` | 3D-RadioDiff | Independent disclosed reimplementation, not official author code. |
| `IDW` | Inverse distance weighting | Runnable non-parametric implementation. |
| `Pix2Pix` | Pix2Pix | Runnable local implementation. |
| `ResNet` | ResNet | Runnable local implementation. |
| `UNet` | U-Net | Runnable local implementation. |
| `PEFNet` | PEFNet-inspired | Runnable building-input VIE-regularized adaptation. It must not be called official PEFNet. |
| `UniRM` | UniRM adapter | Local adapter plus pinned-source setup metadata. Upstream source is not redistributed. |

`3DAFF-RadioDiff` is the proposed method and is therefore not duplicated in
this baseline-only bundle.

## Table values recorded in the manuscript

These numbers are recorded for provenance only. A number in this table must not
be claimed as reproduced by this bundle unless its matching checkpoint,
per-sample metrics, split, and evaluation configuration are retained.

| Method | NMSE | RMSE | PSNR (dB) |
| --- | ---: | ---: | ---: |
| Kriging | 0.0777 | 0.1431 | 23.1646 |
| Kriging + topology | 0.0533 | 0.1201 | 24.8400 |
| 3D-RadioDiff | 0.0840 | 0.1391 | 24.7667 |
| IDW | 0.0340 | 0.0942 | 26.7898 |
| Pix2Pix | 0.0389 | 0.1013 | 26.3333 |
| ResNet | 0.0322 | 0.0897 | 27.3161 |
| U-Net | 0.0273 | 0.0846 | 27.8260 |
| PEFNet | 0.0234 | 0.0780 | 28.6051 |
| UniRM | 0.0247 | 0.0773 | 28.5691 |

The PEFNet and UniRM rows above were obtained with earlier LoS/depth-based
adapters. They are retained only as historical manuscript values and are **not
valid results for the aligned building-input implementations**. Retrain and
replace both rows before claiming a common-input comparison.

## Important academic-use notes

1. Do not describe `RadioDiff3D` as official code. It is a disclosed local
   reimplementation based on the paper and available project materials.
2. Do not describe `PEFNet` as official or exact. The original work does not
   publish all physical quantities and implementation details needed for an
   exact reproduction; the included version is explicitly adapted.
3. The official UniRM repository is public, but the pinned revision does not
   declare a license. Its code is therefore not copied into this package. Run
   the setup command documented under `UniRM` after reviewing upstream terms.
4. The aligned PEFNet and UniRM adapters use `q`, raw building information,
   receiver altitude, and 10% sparse PL; they deliberately ignore NPZ `los`
   and `depth`. Their old checkpoints have different channel semantics and
   cannot be evaluated as aligned models.
5. Before publishing a comparison, use the same train/validation/test split,
   random-10% masks, normalization, measurement-consistency rule, metric region,
   and per-slice aggregation for every method. Do not mix literature numbers
   from another dataset with locally evaluated numbers.

## Primary local data paths

The runnable random-10% baselines currently expect:

```text
D:/datas/processed_pl_aff_dataset/{train,val,test}
D:/zdata/Building_Infomation/buildings
```

Activate the intended environment before using an individual folder:

```powershell
conda activate radiodiffzcy
Set-ExecutionPolicy -Scope Process Bypass
```

Each method folder contains its own README and launch scripts. Read those files
before training because GPU assignment and preprocessing differ by method.

## One-command launchers

### Three-GPU parallel run (recommended)

On the Windows computer with three GPUs, activate `radiodiffzcy` and run:

```powershell
cd "D:\zdata\3DAFF-RadioDiff_Baselines"
conda activate radiodiffzcy
powershell -ExecutionPolicy Bypass -File .\run_3gpu_baselines.ps1 -Action All
```

`All` performs input/output checking, training, and full evaluation. The three
GPU queues start concurrently, and methods assigned to the same GPU run in
sequence:

```text
GPU 0: UNet -> Pix2Pix -> IDW
GPU 1: ResNet -> PEFNet -> Kriging
GPU 2: RadioDiff3D -> UniRM -> KrigingTopology
```

Run only one stage for all methods:

```powershell
# Check model interfaces and data inputs only
powershell -ExecutionPolicy Bypass -File .\run_3gpu_baselines.ps1 -Action Check

# Train learned models only
powershell -ExecutionPolicy Bypass -File .\run_3gpu_baselines.ps1 -Action Train

# Evaluate all methods only
powershell -ExecutionPolicy Bypass -File .\run_3gpu_baselines.ps1 -Action Eval
```

Each GPU queue opens in a separate PowerShell window. Its complete output is
also saved under `D:\zdata\3DAFF-RadioDiff_Baselines\logs`.

All methods use the following common data locations:

```text
D:\datas\processed_pl_aff_dataset\train
D:\datas\processed_pl_aff_dataset\val
D:\datas\processed_pl_aff_dataset\test
D:\zdata\Building_Infomation\buildings
```

### Run one method

The root launcher runs interface checking, training, and full evaluation in the
correct order, including both PEFNet stages:

```powershell
cd "C:\Users\Administrator\Desktop\3DAFF-RadioDiff_Baselines"
.\run_baseline.ps1 -Method UNet -Action All -Gpu 0
```

Replace `UNet` with `ResNet`, `Pix2Pix`, `IDW`, `Kriging`,
`KrigingTopology`, `RadioDiff3D`, `PEFNet`, or `UniRM`. See
`RUN_COMMANDS.md` for every ready-to-run command and the optional sequential
all-method launcher.

## Source links

- UniRM official repository: https://github.com/Shirleyue/UniRM
- PEFNet paper DOI: https://doi.org/10.1109/TWC.2024.3429196
