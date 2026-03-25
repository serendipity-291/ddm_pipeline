# Creates conda env "ddm_pipeline" and installs requirements.txt (PyTorch CPU, MLflow, FastAPI, ...).
# Usage (PowerShell, from repo root):
#   .\setup_env.ps1
# Miniconda/Anaconda must be available (conda init or full path to conda).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    $condaExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($condaExe) {
        function script:conda { & $condaExe @args }
    } else {
        Write-Error "conda not found. Install Miniconda or open Anaconda Prompt."
    }
}

$envName = "ddm_pipeline"
$exists = conda env list | Select-String -Pattern "^\s*$envName\s"
if ($exists) {
    Write-Host "Environment '$envName' already exists. Remove with: conda env remove -n $envName -y" -ForegroundColor Yellow
    exit 1
}

Write-Host "Creating conda env '$envName' (Python 3.11 + pip deps)..." -ForegroundColor Cyan
conda env create -f "$Root\environment.yml"

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "  conda activate $envName"
Write-Host ""
Write-Host "Quick check:" -ForegroundColor Green
Write-Host "  conda run -n $envName python -c `"import torch, mlflow, fastapi; print('torch', torch.__version__)`""
