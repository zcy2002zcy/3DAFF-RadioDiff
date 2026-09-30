param(
    [string]$PythonExecutable = "python",
    [int[]]$Seeds = @(42, 43, 44),
    [int]$DataSeed = 42,
    [switch]$Force,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Invoke-CheckedPython {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

foreach ($seed in $Seeds) {
    $suffix = if ($SmokeTest) { "_smoke" } else { "" }
    $directory = "outputs\pefnet_inspired_adapted_stable_v2_random0_seed${seed}${suffix}"
    $complete = Join-Path $directory "run_complete.json"
    $metrics = Join-Path $directory "evaluation\test\metrics_summary.json"
    $common = @(
        "--config", "configs\pefnet_inspired_adapted.yaml",
        "--seed", "$seed", "--data-seed", "$DataSeed"
    )
    if ($SmokeTest) { $common += "--smoke-test" }

    if ($Force -or -not (Test-Path -LiteralPath $complete)) {
        $trainArguments = @("scripts\train_pefnet_inspired_baseline.py") + $common
        Invoke-CheckedPython -Label "PEFNet-inspired train seed $seed" -Arguments $trainArguments
    } else {
        Write-Host "SKIP completed training: seed $seed" -ForegroundColor DarkGray
    }

    if ($Force -or -not (Test-Path -LiteralPath $metrics)) {
        $evaluationArguments = @("scripts\evaluate_pefnet_inspired_baseline.py") + $common
        Invoke-CheckedPython -Label "PEFNet-inspired evaluate seed $seed" -Arguments $evaluationArguments
    } else {
        Write-Host "SKIP completed evaluation: seed $seed" -ForegroundColor DarkGray
    }
}

Write-Host "`nPEFNet-inspired baseline run completed." -ForegroundColor Green
