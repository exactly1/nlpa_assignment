#!/usr/bin/env pwsh
# setup.ps1 - Create venv and install dependencies on Windows PowerShell

param(
  [string]$Python = "python",
  [string]$VenvPath = ".venv"
)

Write-Host "Creating virtual environment at $VenvPath"
& $Python -m venv $VenvPath

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
Write-Host "Activating virtual environment: $activate"
. $activate

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing backend requirements"
pip install -r backend/requirements.txt

Write-Host "Done. To activate later: . $activate"
