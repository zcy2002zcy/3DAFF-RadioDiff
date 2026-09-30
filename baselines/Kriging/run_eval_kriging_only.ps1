param([int]$Gpu = 2, [int]$MaxSamples = 0, [int]$SubsetSeed = 2025, [string]$PythonExe = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
& $PythonExe scripts/evaluate_kriging.py `
  --config configs/pathloss_kriging_building_random10.yaml `
  --methods kriging `
  --max-samples $MaxSamples `
  --subset-seed $SubsetSeed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
