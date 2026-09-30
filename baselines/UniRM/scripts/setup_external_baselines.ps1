param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Destination = Join-Path $ProjectRoot "external\UniRM"
$Repository = "https://github.com/Shirleyue/UniRM.git"
$Commit = "23d03510b7ae7ef1eeae4195269e2ebf86de0341"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to fetch the pinned UniRM source."
}

if (Test-Path -LiteralPath $Destination) {
    if (-not $Force) {
        throw "Destination already exists: $Destination. Use -Force only to refresh it."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Destination ".git"))) {
        throw "Refusing to replace a non-Git directory: $Destination"
    }
    git -C $Destination fetch --all --tags
} else {
    git clone $Repository $Destination
}
if ($LASTEXITCODE -ne 0) { throw "Failed to fetch UniRM." }

git -C $Destination checkout --detach $Commit
if ($LASTEXITCODE -ne 0) { throw "Failed to check out pinned UniRM commit $Commit." }

$Actual = (git -C $Destination rev-parse HEAD).Trim()
if ($Actual -ne $Commit) { throw "UniRM commit mismatch: expected $Commit, got $Actual" }
Write-Host "UniRM ready at pinned commit $Actual" -ForegroundColor Green
