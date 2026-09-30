param(
    [Parameter(Mandatory = $true)]
    [int]$Gpu,
    [Parameter(Mandatory = $true)]
    [string]$MethodsCsv,
    [ValidateSet("Check", "Train", "Eval", "All")]
    [string]$Action = "All",
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $Root "run_baseline.ps1"
$logDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "gpu${Gpu}_${Action}_${stamp}.log"
$methods = $MethodsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)

Start-Transcript -LiteralPath $logPath -Force | Out-Null
try {
    Write-Host "GPU $Gpu queue: $($methods -join ' -> ')" -ForegroundColor Cyan
    Write-Host "Python: $PythonExe" -ForegroundColor Cyan
    Write-Host "Log: $logPath" -ForegroundColor Cyan

    foreach ($method in $methods) {
        Write-Host "`n===== GPU $Gpu | $method | $Action =====" -ForegroundColor Yellow
        & powershell -ExecutionPolicy Bypass -File $runner `
            -Method $method -Action $Action -Gpu $Gpu `
            -PythonExe $PythonExe -ScientificPythonExe $PythonExe
        if ($LASTEXITCODE -ne 0) {
            throw "$method failed on physical GPU $Gpu with exit code $LASTEXITCODE"
        }
    }

    Write-Host "GPU $Gpu queue completed." -ForegroundColor Green
}
catch {
    Write-Host "GPU $Gpu queue failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}

python -c "import sys; print(sys.executable); print(sys.prefix)"