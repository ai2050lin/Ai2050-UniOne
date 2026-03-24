$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvActivate)) {
    throw "未找到虚拟环境激活脚本: $venvActivate"
}

if (-not (Test-Path $venvPython)) {
    throw "未找到虚拟环境 Python: $venvPython"
}

Set-Location $projectRoot

Write-Host "正在进入 .venv 虚拟环境..." -ForegroundColor Cyan
. $venvActivate

Write-Host "当前 Python: $venvPython" -ForegroundColor Green
Write-Host "正在启动后端服务 server/server.py ..." -ForegroundColor Cyan

& $venvPython "server\server.py"
