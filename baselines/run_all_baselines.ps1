param(
    [ValidateSet("Check", "Train", "Eval", "All")]
    [string]$Action = "All"
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_baseline.ps1"
$jobs = @(
    @{ Method = "UNet"; Gpu = 0 },
    @{ Method = "ResNet"; Gpu = 1 },
    @{ Method = "Pix2Pix"; Gpu = 0 },
    @{ Method = "IDW"; Gpu = 1 },
    @{ Method = "Kriging"; Gpu = 2 },
    @{ Method = "KrigingTopology"; Gpu = 2 },
    @{ Method = "RadioDiff3D"; Gpu = 0 },
    @{ Method = "PEFNet"; Gpu = 1 },
    @{ Method = "UniRM"; Gpu = 2 }
)

foreach ($job in $jobs) {
    & powershell -ExecutionPolicy Bypass -File $runner `
        -Method $job.Method -Action $Action -Gpu $job.Gpu
    if ($LASTEXITCODE -ne 0) { throw "$($job.Method) failed" }
}

Write-Host "All baseline jobs completed." -ForegroundColor Green
