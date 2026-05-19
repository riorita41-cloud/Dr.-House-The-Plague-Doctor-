#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ДОКТОР ХАУС — Чумной Доктор" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Create venv if not exists
if (-not (Test-Path "venv")) {
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
& .\venv\Scripts\Activate.ps1

# Install deps if missing
if (-not (Test-Path "venv\.installed")) {
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    pip install -q -r requirements.txt
    New-Item -ItemType File -Path "venv\.installed" -Force | Out-Null
}

# Train model if not present
if (-not (Test-Path "doctor_house_model_v8.pkl")) {
    Write-Host "[*] Training model..." -ForegroundColor Yellow
    python step1_train_v8_final.py
}

Write-Host "[*] Starting server..." -ForegroundColor Green
Write-Host "    Open http://127.0.0.1:5000 in browser" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
python server.py
