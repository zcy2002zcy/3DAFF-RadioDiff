param([int]$Gpu = 0, [int]$MaxVolumes = 100, [switch]$Full)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }
if ($Full) { $MaxVolumes = 0 }
$arguments = @(
    "-u", "scripts/evaluate.py",
    "--config", "configs/pl_random10_standard.yaml",
    "--max-volumes", "$MaxVolumes",
    "--consistency-mode", "projection_each_step",
    "--output-dir", "outputs/radiodiff3d_pl_random10_standard/eval_results_stepwise"
)
& $PythonExe @arguments
