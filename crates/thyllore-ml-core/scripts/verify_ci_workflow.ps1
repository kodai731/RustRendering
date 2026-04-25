# Simulates the GitHub Actions python_parity.yml workflow locally.
#
# Purpose:
#   Detect mismatches between local maturin develop (venv-active) and CI
#   maturin build + pip install (no-venv) flows BEFORE pushing.
#
# Usage (from anywhere):
#   pwsh crates/thyllore-ml-core/scripts/verify_ci_workflow.ps1
#
# What this script does (mirrors .github/workflows/python_parity.yml):
#   1. cargo test --test parity_fixtures generate_parity_fixtures
#   2. maturin build --release --features python --out dist
#   3. Create a fresh ephemeral venv (NOT the developer's .venv)
#   4. pip install pytest numpy
#   5. pip install --force-reinstall <generated wheel>
#   6. pytest tests/python_parity/ -v
#
# Exit code 0 = all CI steps reproducible locally.
# Exit code != 0 = a step failed; the same step would fail in GitHub Actions.

[CmdletBinding()]
param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$crateDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$tempVenv = Join-Path $env:TEMP "thyllore-ml-core-ci-verify-$(Get-Random)"

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Invoke-Checked($description, [scriptblock]$action) {
    & $action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $description (exit code $LASTEXITCODE)" -ForegroundColor Red
        throw "$description failed"
    }
}

Push-Location $crateDir
try {
    Write-Step "Step 1/6: Generate Rust fixtures"
    Invoke-Checked "cargo test --test parity_fixtures" {
        cargo test --test parity_fixtures generate_parity_fixtures
    }

    Write-Step "Step 2/6: Build wheel with maturin"
    if (Test-Path "dist") {
        Remove-Item dist -Recurse -Force
    }
    Invoke-Checked "maturin build" {
        maturin build --release --features python --out dist
    }

    $wheelFiles = Get-ChildItem dist/*.whl
    if ($wheelFiles.Count -eq 0) {
        throw "No wheel produced under dist/"
    }
    $wheelPath = $wheelFiles[0].FullName
    Write-Host "Built wheel: $wheelPath"

    Write-Step "Step 3/6: Create ephemeral venv (NOT the dev .venv)"
    Invoke-Checked "python -m venv (ephemeral)" {
        & $PythonExe -m venv $tempVenv
    }
    $venvPython = Join-Path $tempVenv "Scripts\python.exe"
    $venvPip = Join-Path $tempVenv "Scripts\pip.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Ephemeral venv python not found at $venvPython"
    }

    Write-Step "Step 4/6: pip install pytest numpy (ephemeral venv)"
    Invoke-Checked "pip install deps" {
        & $venvPip install --quiet pytest "numpy>=1.21,<3.0"
    }

    Write-Step "Step 5/6: pip install wheel (ephemeral venv, simulating CI)"
    Invoke-Checked "pip install wheel" {
        & $venvPip install --quiet --force-reinstall $wheelPath
    }

    Write-Step "Step 6/6: pytest python_parity (ephemeral venv)"
    Invoke-Checked "pytest" {
        & $venvPython -m pytest tests/python_parity/ -v
    }

    Write-Host ""
    Write-Host "All CI workflow steps passed locally" -ForegroundColor Green
    Write-Host "(.github/workflows/python_parity.yml should succeed for this branch)" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "CI workflow simulation FAILED: $_" -ForegroundColor Red
    Write-Host "(.github/workflows/python_parity.yml will likely fail with the same error)" -ForegroundColor Red
    exit 1
} finally {
    if (Test-Path $tempVenv) {
        Remove-Item $tempVenv -Recurse -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
