$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot

$env:PYTHONUTF8 = '1'
$env:UV_CACHE_DIR = Join-Path $workspaceRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $workspaceRoot '.python'
$env:UV_PYTHON_BIN_DIR = Join-Path $workspaceRoot '.python-bin'
$env:TEMP = Join-Path $workspaceRoot '.tmp'
$env:TMP = $env:TEMP

. (Join-Path $projectRoot '.venv\Scripts\Activate.ps1')
Set-Location $projectRoot

Write-Host "ALPHASE environment activated."
Write-Host "Python: $((Get-Command python).Source)"
Write-Host "Data:   $(Join-Path $workspaceRoot 'data')"
