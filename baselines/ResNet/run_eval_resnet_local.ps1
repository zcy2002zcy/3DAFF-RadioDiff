param([int]$Gpu = 1, [string]$PythonExe = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
& $PythonExe scripts/evaluate_resnet.py --config configs/pathloss_resnet_building_random10.yaml --checkpoint best
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
