# One-command baseline runs

Run from:

```powershell
cd "C:\Users\Administrator\Desktop\3DAFF-RadioDiff_Baselines"
Set-ExecutionPolicy -Scope Process Bypass
```

Each `All` command performs check, training, and full evaluation in order:

```powershell
.\run_baseline.ps1 -Method UNet             -Action All -Gpu 0
.\run_baseline.ps1 -Method ResNet           -Action All -Gpu 1
.\run_baseline.ps1 -Method Pix2Pix          -Action All -Gpu 0
.\run_baseline.ps1 -Method IDW              -Action All -Gpu 1
.\run_baseline.ps1 -Method Kriging          -Action All -Gpu 2
.\run_baseline.ps1 -Method KrigingTopology  -Action All -Gpu 2
.\run_baseline.ps1 -Method RadioDiff3D      -Action All -Gpu 0
.\run_baseline.ps1 -Method PEFNet           -Action All -Gpu 1
.\run_baseline.ps1 -Method UniRM            -Action All -Gpu 2
```

Run only one phase by replacing `All` with `Check`, `Train`, or `Eval`:

```powershell
.\run_baseline.ps1 -Method UNet -Action Check -Gpu 0
.\run_baseline.ps1 -Method UNet -Action Train -Gpu 0
.\run_baseline.ps1 -Method UNet -Action Eval  -Gpu 0
```

Sequentially run every method (this can take several days):

```powershell
.\run_all_baselines.ps1 -Action All
```

Check all interfaces without training:

```powershell
.\run_all_baselines.ps1 -Action Check
```

## Three-GPU parallel run with radiodiffzcy

Activate the environment, then start three concurrent queues. Each GPU runs
its assigned methods sequentially, while GPU 0, 1, and 2 work in parallel.

```powershell
cd "D:\zdata\3DAFF-RadioDiff_Baselines"
conda activate radiodiffzcy
Set-ExecutionPolicy -Scope Process Bypass
.\run_3gpu_baselines.ps1 -Action All
```

Assignments:

```text
GPU 0: UNet -> Pix2Pix -> IDW
GPU 1: ResNet -> PEFNet -> Kriging
GPU 2: RadioDiff3D -> UniRM -> KrigingTopology
```

Only check all interfaces without training:

```powershell
.\run_3gpu_baselines.ps1 -Action Check
```

Print the assignments and validate the active Python without launching jobs:

```powershell
.\run_3gpu_baselines.ps1 -Action All -DryRun
```

Each queue opens in its own PowerShell window. Complete terminal output is
also stored under `logs\gpu0_*`, `logs\gpu1_*`, and `logs\gpu2_*`.

All learned methods use the common random-10% path-loss dataset and full
40,000-slice evaluation. The publication primary metrics are full-slice NMSE,
RMSE, SSIM, and PSNR after final measurement projection. Unobserved outdoor
metrics remain secondary diagnostics.

The launcher uses `python.exe` from the active `radiodiffzcy` environment.
An explicit interpreter can still be supplied when needed:

```powershell
.\run_baseline.ps1 -Method UNet -Action All -Gpu 0 `
  -PythonExe "D:\Anaconda3\envs\radiodiffzcy\python.exe"
```
