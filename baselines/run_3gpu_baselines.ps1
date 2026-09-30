param(
    [ValidateSet("Check", "Train", "Eval", "All")]
    [string]$Action = "All",
    [string]$PythonExe = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$worker = Join-Path $Root "run_gpu_queue.ps1"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $targetPrefix = $null
    try {
        $condaInfo = (& conda env list --json 2>$null | Out-String | ConvertFrom-Json)
        $targetPrefix = $condaInfo.envs | Where-Object {
            (Split-Path -Leaf $_) -eq "radiodiffzcy"
        } | Select-Object -First 1
    }
    catch {
        $targetPrefix = $null
    }

    if ([string]::IsNullOrWhiteSpace($targetPrefix)) {
        $knownCandidates = @(
            "D:\Anaconda3\envs\radiodiffzcy",
            "C:\Anaconda3\envs\radiodiffzcy",
            "$env:USERPROFILE\.conda\envs\radiodiffzcy"
        )
        $targetPrefix = $knownCandidates | Where-Object {
            Test-Path -LiteralPath (Join-Path $_ "python.exe") -PathType Leaf
        } | Select-Object -First 1
    }

    if ([string]::IsNullOrWhiteSpace($targetPrefix)) {
        throw "Cannot locate the radiodiffzcy Conda environment. Run: conda env list"
    }
    $PythonExe = Join-Path $targetPrefix "python.exe"
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
$pythonPrefix = (& $PythonExe -c "import sys; print(sys.prefix)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonPrefix)) {
    throw "Unable to query the selected Python environment: $PythonExe"
}
$pythonEnvironment = Split-Path -Leaf $pythonPrefix
if ($pythonEnvironment -ne "radiodiffzcy") {
    throw "Wrong Python environment: $pythonPrefix. Activate radiodiffzcy first."
}
if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) {
    throw "GPU worker script not found: $worker"
}

$queues = @(
    @{ Gpu = 0; Methods = "UNet,Pix2Pix,IDW" },
    @{ Gpu = 1; Methods = "ResNet,PEFNet,Kriging" },
    @{ Gpu = 2; Methods = "RadioDiff3D,UniRM,KrigingTopology" }
)

Write-Host "Verified environment: $pythonEnvironment" -ForegroundColor Cyan
Write-Host "Python: $PythonExe" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Cyan
foreach ($queue in $queues) {
    Write-Host "GPU $($queue.Gpu): $($queue.Methods.Replace(',', ' -> '))"
}

if ($DryRun) {
    Write-Host "DRY RUN PASSED: no baseline process was started." -ForegroundColor Green
    exit 0
}

$processes = @()
foreach ($queue in $queues) {
    $arguments = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $worker,
        "-Gpu", "$($queue.Gpu)",
        "-MethodsCsv", $queue.Methods,
        "-Action", $Action,
        "-PythonExe", $PythonExe,
        "-Root", $Root
    )
    $processes += Start-Process -FilePath "powershell.exe" `
        -ArgumentList $arguments -WorkingDirectory $Root -PassThru
}

Write-Host "Three GPU queues started. Waiting for completion..." -ForegroundColor Green
$processes | Wait-Process

$failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
    $ids = ($failed | ForEach-Object { $_.Id }) -join ", "
    throw "One or more GPU queues failed. Process IDs: $ids. Check the logs directory."
}

Write-Host "All three GPU queues completed successfully." -ForegroundColor Green
