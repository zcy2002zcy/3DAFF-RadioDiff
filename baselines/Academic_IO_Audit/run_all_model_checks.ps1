param(
    [int]$Gpu = 0,
    [string]$DatasetRoot = "D:\datas\processed_pl_aff_dataset",
    [string]$ProposedProject = "C:\Users\Administrator\Desktop\3DAFF-RadioDiff-PathLoss",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$AuditRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Split-Path -Parent $AuditRoot
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = "python" }
$pythonDir = Split-Path -Parent $PythonExe
if (-not [string]::IsNullOrWhiteSpace($pythonDir) -and (Test-Path -LiteralPath $pythonDir)) {
    $env:PATH = "$pythonDir;$env:PATH"
}

Write-Host "[1/10] Common dataset and leakage audit" -ForegroundColor Cyan
& $PythonExe "$AuditRoot\audit_io_contract.py" --dataset-root $DatasetRoot --sample-rate 0.10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$checks = @(
    @{ Name = "U-Net"; Dir = "UNet"; Script = "run_check_unet_local.ps1"; UsesGpu = $true; AcceptsPythonExe = $true },
    @{ Name = "ResNet"; Dir = "ResNet"; Script = "run_check_resnet_local.ps1"; UsesGpu = $true; AcceptsPythonExe = $true },
    @{ Name = "Pix2Pix"; Dir = "Pix2Pix"; Script = "run_check_p2p_local.ps1"; UsesGpu = $true; AcceptsPythonExe = $true },
    @{ Name = "IDW"; Dir = "IDW"; Script = "run_check_idw_local.ps1"; UsesGpu = $false; AcceptsPythonExe = $true },
    @{ Name = "Kriging"; Dir = "Kriging"; Script = "run_check_kriging.ps1"; UsesGpu = $true; AcceptsPythonExe = $true },
    @{ Name = "3D-RadioDiff reimplementation"; Dir = "RadioDiff3D"; Script = "run_check_standard.ps1"; UsesGpu = $true; AcceptsPythonExe = $false }
)

$index = 2
foreach ($check in $checks) {
    Write-Host "[$index/10] $($check.Name) model-forward check" -ForegroundColor Cyan
    $project = Join-Path $BundleRoot $check.Dir
    $launcher = Join-Path $project $check.Script
    if (-not (Test-Path -LiteralPath $launcher)) {
        throw "Missing launcher: $launcher"
    }
    if ($check.Name -eq "3D-RadioDiff reimplementation") {
        $manifest = Join-Path $project "cache\manifests\test.json"
        if (-not (Test-Path -LiteralPath $manifest)) {
            Write-Host "3D-RadioDiff status: WAITING_FOR_MANIFEST" -ForegroundColor Yellow
            Write-Host "Run $project\run_prepare.ps1 before its model-forward check." -ForegroundColor Yellow
            $index++
            continue
        }
    }
    Push-Location $project
    try {
        if ($check.UsesGpu -and $check.AcceptsPythonExe) {
            & powershell -ExecutionPolicy Bypass -File $launcher -Gpu $Gpu -PythonExe $PythonExe
        }
        elseif ($check.UsesGpu) {
            & powershell -ExecutionPolicy Bypass -File $launcher -Gpu $Gpu
        }
        elseif ($check.AcceptsPythonExe) {
            & powershell -ExecutionPolicy Bypass -File $launcher -PythonExe $PythonExe
        }
        else {
            & powershell -ExecutionPolicy Bypass -File $launcher
        }
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }
    $index++
}

Write-Host "[8/10] UniRM adapter model-forward check" -ForegroundColor Cyan
& $PythonExe "D:\zdata\UniRM-baseline\check_building_io.py" --output "$AuditRoot\unirm_building_io_report.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[9/10] PEFNet-inspired aligned model-forward check" -ForegroundColor Cyan
& $PythonExe "C:\Users\Administrator\Documents\idea1\PEFNet-baseline\check_building_io.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item -LiteralPath "C:\Users\Administrator\Documents\idea1\PEFNet-baseline\pefnet_building_io_report.json" -Destination "$AuditRoot\pefnet_building_io_report.json" -Force

Write-Host "[10/10] 3DAFF-RadioDiff proposed-model check" -ForegroundColor Cyan
$proposedLauncher = Join-Path $ProposedProject "run_check.ps1"
if (-not (Test-Path -LiteralPath $proposedLauncher)) {
    throw "Missing proposed-model launcher: $proposedLauncher"
}
Push-Location $ProposedProject
try {
    & powershell -ExecutionPolicy Bypass -File $proposedLauncher -Gpu $Gpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

Write-Host "UniRM and PEFNet-inspired inputs are aligned, but both require retraining because old checkpoint channel semantics differ." -ForegroundColor Yellow
Write-Host "AUDIT RUN COMPLETED. Review every PASS/WARNING/BLOCKED status above." -ForegroundColor Green
