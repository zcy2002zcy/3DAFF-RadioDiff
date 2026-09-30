param([string]$PythonExe = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
& $PythonExe scripts/train_idw.py --config configs/pathloss_idw_random10.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
