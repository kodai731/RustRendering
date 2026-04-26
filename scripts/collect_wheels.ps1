[CmdletBinding()]
param(
    [string]$WheelsDir = "blender_addon/wheels",
    [switch]$SkipMaturin,
    [switch]$SkipPipDownload,
    [string[]]$Platforms = @("win_amd64", "manylinux2014_x86_64", "macosx_11_0_arm64"),
    [string]$GrpcioVersion = "1.71.2",
    [string]$ProtobufVersion = "5.29.6",
    [string]$CertifiVersion = "2024.12.14"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."
$AbsWheels = Join-Path $RepoRoot $WheelsDir

New-Item -ItemType Directory -Path $AbsWheels -Force | Out-Null

if (-not $SkipPipDownload) {
    Write-Host "[collect_wheels] Downloading vendored runtime wheels..." -ForegroundColor Cyan
    # pip writes status to stderr; tolerate that without surfacing as an error
    $PrevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        foreach ($platform in $Platforms) {
            Write-Host "  Platform: $platform"
            $pipArgs = @(
                "-m", "pip", "download",
                "grpcio==$GrpcioVersion",
                "grpcio-status==$GrpcioVersion",
                "protobuf==$ProtobufVersion",
                "certifi==$CertifiVersion",
                "--platform", $platform,
                "--abi", "cp310",
                "--implementation", "cp",
                "--only-binary=:all:",
                "--dest", $AbsWheels,
                "--no-deps"
            )
            $output = & python @pipArgs 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host ($output -join "`n")
                throw "pip download failed for $platform"
            }
        }
    } finally {
        $ErrorActionPreference = $PrevErrorAction
    }

    # certifi and grpcio_status are pure Python (py3-none-any) — pip downloads
    # the same file three times, which is harmless but wastes one download.
    # The duplicates are deduplicated automatically because the filename is
    # identical.
}

if (-not $SkipMaturin) {
    Write-Host "[collect_wheels] Building thyllore_ml_core wheel via maturin..." -ForegroundColor Cyan
    & python -m pip install --quiet maturin *>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "maturin install failed"
    }

    Push-Location (Join-Path $RepoRoot "crates/thyllore-ml-core")
    try {
        # maturin build defaults to the host platform. For full 3-platform
        # coverage CI runs this script on each platform's runner, then
        # collect_wheels.ps1 -SkipPipDownload to assemble the final dir.
        & maturin build --release --features python --out $AbsWheels *>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "maturin build failed"
        }
    } finally {
        Pop-Location
    }
}

# SHA256 manifest written for local debugging only — gitignored.
# Phase 6 will replace this with `pip --require-hashes` on a committed
# requirements lock file (see parent design doc).
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
