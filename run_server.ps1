param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [ValidateSet('check','preflight','train','all','calibrate','eval')][string]$Action='check',
    [ValidateRange(0,32)][int]$Gpu=0,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
$Runner=Join-Path $PSScriptRoot 'run.py'
$RunnerArgs=@('-B','-u',$Runner,'--action',$Action,'--dataset-root',$DatasetRoot,'--gpu',"$Gpu")
if ($Resume) { $RunnerArgs+='--resume' }
& python @RunnerArgs
if ($LASTEXITCODE -ne 0) { throw 'Action failed. No automatic retry or output cleanup.' }
