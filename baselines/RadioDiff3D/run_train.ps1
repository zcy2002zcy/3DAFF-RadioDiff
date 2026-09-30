param([int]$Gpu = 0, [switch]$Resume)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$argsList = @("-u", "scripts/train.py", "--config", "configs/pl_random10.yaml")
if ($Resume) { $argsList += "--resume" }
python @argsList
