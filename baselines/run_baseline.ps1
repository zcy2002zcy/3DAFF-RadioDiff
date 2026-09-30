param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("UNet", "ResNet", "Pix2Pix", "IDW", "Kriging", "KrigingTopology", "RadioDiff3D", "PEFNet", "UniRM")]
    [string]$Method,
    [ValidateSet("Check", "Train", "Eval", "All")]
    [string]$Action = "All",
    [int]$Gpu = 0,
    [string]$PythonExe = "",
    [string]$ScientificPythonExe = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "python.exe is not available on PATH. Activate radiodiffzcy first."
    }
    $PythonExe = $pythonCommand.Source
}
if ([string]::IsNullOrWhiteSpace($ScientificPythonExe)) {
    $ScientificPythonExe = $PythonExe
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $ScientificPythonExe -PathType Leaf)) {
    throw "Scientific Python executable not found: $ScientificPythonExe"
}

function Invoke-Launcher {
    param([string]$Path, [string[]]$Arguments = @())
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing launcher: $Path" }
    & powershell -ExecutionPolicy Bypass -File $Path @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Launcher failed: $Path" }
}

function Wants([string]$Name) {
    return $Action -eq $Name -or $Action -eq "All"
}

Write-Host "Method=$Method Action=$Action PhysicalGPU=$Gpu" -ForegroundColor Cyan

switch ($Method) {
    "UNet" {
        $dir = Join-Path $Root "UNet"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_unet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Train") { Invoke-Launcher "$dir\run_train_unet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_unet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
    }
    "ResNet" {
        $dir = Join-Path $Root "ResNet"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_resnet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Train") { Invoke-Launcher "$dir\run_train_resnet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_resnet_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
    }
    "Pix2Pix" {
        $dir = Join-Path $Root "Pix2Pix"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_p2p_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Train") { Invoke-Launcher "$dir\run_train_p2p_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_p2p_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
    }
    "IDW" {
        $dir = Join-Path $Root "IDW"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_idw_local.ps1" @("-PythonExe", $PythonExe) }
        if (Wants "Train") { Invoke-Launcher "$dir\run_train_idw_local.ps1" @("-PythonExe", $PythonExe) }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_idw_local.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
    }
    "Kriging" {
        $dir = Join-Path $Root "Kriging"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_kriging.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Train") { Write-Host "Kriging is non-parametric; no training stage." -ForegroundColor Yellow }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_kriging_only.ps1" @("-Gpu", "$Gpu", "-MaxSamples", "0", "-PythonExe", $PythonExe) }
    }
    "KrigingTopology" {
        $dir = Join-Path $Root "Kriging"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_kriging.ps1" @("-Gpu", "$Gpu", "-PythonExe", $PythonExe) }
        if (Wants "Train") { Write-Host "Kriging + topology is non-parametric; no training stage." -ForegroundColor Yellow }
        if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_kriging_with_topology.ps1" @("-Gpu", "$Gpu", "-MaxSamples", "0", "-PythonExe", $PythonExe) }
    }
    "RadioDiff3D" {
        $dir = Join-Path $Root "RadioDiff3D"
        $oldPath = $env:PATH
        $env:PATH = "$(Split-Path -Parent $PythonExe);$oldPath"
        try {
            if (-not (Test-Path -LiteralPath "$dir\cache\manifests\test.json")) {
                Invoke-Launcher "$dir\run_prepare.ps1" @("-Workers", "4")
            }
            if (Wants "Check") { Invoke-Launcher "$dir\run_check_standard.ps1" @("-Gpu", "$Gpu") }
            if (Wants "Train") { Invoke-Launcher "$dir\run_train_standard.ps1" @("-Gpu", "$Gpu") }
            if (Wants "Eval")  { Invoke-Launcher "$dir\run_eval_standard.ps1" @("-Gpu", "$Gpu", "-MaxVolumes", "0") }
        }
        finally { $env:PATH = $oldPath }
    }
    "PEFNet" {
        $dir = Join-Path $Root "PEFNet"
        if (Wants "Check") { Invoke-Launcher "$dir\run_check_building_random10.ps1" @("-PythonExe", $ScientificPythonExe) }
        if (Wants "Train") {
            Invoke-Launcher "$dir\run_train_building_random10.ps1" @("-Stage", "physics", "-Gpu", "$Gpu", "-PythonExe", $ScientificPythonExe)
            Invoke-Launcher "$dir\run_train_building_random10.ps1" @("-Stage", "refine", "-Gpu", "$Gpu", "-PythonExe", $ScientificPythonExe)
        }
        if (Wants "Eval") { Invoke-Launcher "$dir\run_eval_building_random10.ps1" @("-Gpu", "$Gpu", "-PythonExe", $ScientificPythonExe) }
    }
    "UniRM" {
        $dir = "D:\zdata\UniRM-baseline"
        if (-not (Test-Path -LiteralPath $dir)) { throw "Missing UniRM adapter: $dir" }
        $env:CUDA_VISIBLE_DEVICES = "$Gpu"
        Push-Location $dir
        try {
            if (Wants "Check") {
                & $ScientificPythonExe .\check_building_io.py
                if ($LASTEXITCODE -ne 0) { throw "UniRM check failed" }
            }
            if (Wants "Train") {
                & $ScientificPythonExe .\train_pathloss.py `
                    --data-root D:\datas\processed_pl_aff_dataset `
                    --building-dir D:\zdata\Building_Infomation\buildings `
                    --sampling-mode random --sampling-rate 0.10 `
                    --epochs 10 --batch-size 2 --num-workers 4 `
                    --learning-rate 5e-5 --enable-prompt 1 `
                    --device cuda --output-dir runs\unirm_building_random10
                if ($LASTEXITCODE -ne 0) { throw "UniRM training failed" }
            }
            if (Wants "Eval") {
                & $ScientificPythonExe .\eval.py `
                    --checkpoint runs\unirm_building_random10\best.pt `
                    --data-root D:\datas\processed_pl_aff_dataset `
                    --building-dir D:\zdata\Building_Infomation\buildings `
                    --test-limit 0 --device cuda
                if ($LASTEXITCODE -ne 0) { throw "UniRM evaluation failed" }
            }
        }
        finally { Pop-Location }
    }
}

Write-Host "$Method $Action completed." -ForegroundColor Green
