param(
    [int]$Gpu = 0,
    [string]$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
& $PythonExe .\eval.py `
    --checkpoint runs\pefnet_building_random10_refine\best.pt `
    --data-root D:\datas\processed_pl_aff_dataset `
    --building-dir D:\zdata\Building_Infomation\buildings `
    --test-limit 0 --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
