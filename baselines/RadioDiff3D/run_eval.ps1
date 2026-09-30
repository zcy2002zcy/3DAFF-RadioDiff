param([int]$Gpu = 0, [int]$MaxVolumes = 0, [switch]$NoConsistency)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$argsList = @("-u", "scripts/evaluate.py", "--config", "configs/pl_random10.yaml", "--max-volumes", "$MaxVolumes")
if ($NoConsistency) { $argsList += "--no-consistency" }
python @argsList
