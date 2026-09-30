param([int]$Gpu = 1, [string]$PythonExe = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
& $PythonExe scripts/evaluate_idw.py --config configs/pathloss_idw_random10.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
