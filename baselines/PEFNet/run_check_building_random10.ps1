param(
    [string]$PythonExe = "D:\Anaconda3\envs\rmdm5070\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& $PythonExe .\check_building_io.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
