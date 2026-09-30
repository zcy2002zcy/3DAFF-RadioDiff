param([int]$Gpu = 0, [switch]$Resume)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
Set-Location $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$arguments = @("-u", "scripts/train.py", "--config", "configs/pl_random10_standard.yaml")
if ($Resume) { $arguments += "--resume" }
python @arguments
