param([int]$Workers = 4, [switch]$Force)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRoot
$argsList = @("scripts/build_manifests.py", "--config", "configs/pl_random10.yaml", "--workers", "$Workers")
if ($Force) { $argsList += "--force" }
python @argsList
