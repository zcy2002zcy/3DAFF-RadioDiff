param([int]$Gpu = 0, [string]$PythonExe = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
& $PythonExe scripts/evaluate_unet.py --config configs/pathloss_unet_building_random10.yaml --checkpoint best
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
