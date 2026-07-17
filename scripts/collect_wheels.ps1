[CmdletBinding()]
param(
    [string]$WheelsDir = "blender_addon/wheels",
    [switch]$SkipMaturin
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."
$AbsWheels = Join-Path $RepoRoot $WheelsDir

New-Item -ItemType Directory -Path $AbsWheels -Force | Out-Null

if (-not $SkipMaturin) {
    Write-Host "[collect_wheels] Building thyllore_ml_core wheel via maturin..." -ForegroundColor Cyan
    $PrevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -m pip install --quiet maturin *>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "maturin install failed"
        }

        Push-Location (Join-Path $RepoRoot "crates/thyllore-ml-core")
        $PrevRustFlags = $env:RUSTFLAGS
        try {
            $RemapFlag = "--remap-path-prefix=$HOME=."
            $env:RUSTFLAGS = if ($PrevRustFlags) { "$PrevRustFlags $RemapFlag" } else { $RemapFlag }
            & maturin build --release --features python --out $AbsWheels *>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "maturin build failed"
            }
        } finally {
            $env:RUSTFLAGS = $PrevRustFlags
            Pop-Location
        }
    } finally {
        $ErrorActionPreference = $PrevErrorAction
    }
}

# SHA256 manifest written for local debugging only — gitignored.
Write-Host "[collect_wheels] Writing local SHA256 manifest (gitignored)..." -ForegroundColor Cyan
$HashesFile = Join-Path $AbsWheels "HASHES.txt"
$Lines = New-Object System.Collections.Generic.List[string]
foreach ($wheel in Get-ChildItem $AbsWheels -Filter "*.whl" | Sort-Object Name) {
    $hash = (Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower()
    $Lines.Add("$($wheel.Name)  $hash")
}
[System.IO.File]::WriteAllText($HashesFile, ($Lines -join "`n") + "`n")

$WheelCount = (Get-ChildItem $AbsWheels -Filter "*.whl").Count
Write-Host "[collect_wheels] Done. $WheelCount wheels in $AbsWheels"
