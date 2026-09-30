param(
    [ValidateSet("physics", "refine")][string]$Stage = "physics",
    [int]$Gpu = 0,
    [string]$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$common = @(
    "--data-root", "D:\datas\processed_pl_aff_dataset",
    "--building-dir", "D:\zdata\Building_Infomation\buildings",
    "--sampling-mode", "random", "--sampling-rate", "0.10",
    "--epochs", "10", "--batch-size", "2", "--num-workers", "4",
    "--learning-rate", "1e-4", "--device", "cuda"
)
if ($Stage -eq "physics") {
    & $PythonExe .\train.py @common --stage physics --output-dir runs\pefnet_building_random10_physics
} else {
    & $PythonExe .\train.py @common --stage refine `
        --init-checkpoint runs\pefnet_building_random10_physics\best.pt `
        --output-dir runs\pefnet_building_random10_refine
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
