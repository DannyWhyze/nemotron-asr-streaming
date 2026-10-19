# Nemotron 3.5 ASR STT Server - CPU (AVX2) Launcher
$Host.UI.RawUI.WindowTitle = "Nemotron 3.5 ASR Server (CPU - AVX2)"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting Nemotron 3.5 ASR Server on CPU (AVX2)..." -ForegroundColor Yellow
Write-Host "  Directory: $ProjectRoot" -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "Error: Virtual environment python.exe not found at $PythonExe" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

& $PythonExe run_server.py --backend cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nServer terminated with code $LASTEXITCODE." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
}
