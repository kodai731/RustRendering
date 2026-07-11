[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("win_amd64", "linux_x86_64", "macosx_arm64")]
    [string]$Platform,

    # MVP "lite" Variant ships only Tier B (curve_copilot, and once the
    # light_t2m PyO3 rewrite lands, text_to_motion). The "full" Variant adds
    # the Tier A gRPC operators (auto_rig / text_to_mesh) and is intended for
    # the Phase 6 SaaS-authenticated deployment.
    [ValidateSet("lite", "full")]
    [string]$Variant = "lite",

    # Distribution build mode (see the curve copilot data collection design):
    # A: official repo, no telemetry, ctx32 fixed
    # B: self-hosted static repo, telemetry opt-in (requires
    #    THYLLORE_FEEDBACK_ENDPOINT, THYLLORE_INGEST_TOKEN,
    #    THYLLORE_UNLOCK_PUBKEY_B64)
    # C: Blender Market, online license activation (requires
    #    THYLLORE_LICENSE_ENDPOINT, THYLLORE_UNLOCK_PUBKEY_B64)
    [ValidateSet("A", "B", "C")]
    [string]$BuildMode = "A",

    [string]$Version = "0.0.1",

    [string]$OutputDir = "dist",

    [switch]$IncludeOnnxModel,

    [string]$OnnxSourcePath = "ml/model/curve_copilot.onnx",

    [switch]$SkipBlenderValidate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."

if (-not $PSBoundParameters.ContainsKey('OnnxSourcePath') -and $env:THYLLORE_CURVE_COPILOT_ONNX) {
    $OnnxSourcePath = $env:THYLLORE_CURVE_COPILOT_ONNX
}

$FeedbackEndpoint = $env:THYLLORE_FEEDBACK_ENDPOINT
$IngestToken = $env:THYLLORE_INGEST_TOKEN
$UnlockPubkey = $env:THYLLORE_UNLOCK_PUBKEY_B64
$LicenseEndpoint = $env:THYLLORE_LICENSE_ENDPOINT

function Assert-BuildModeEnv([string]$Name, [string]$Value) {
    if ([string]::IsNullOrEmpty($Value)) {
        throw "build mode $BuildMode requires environment variable $Name"
    }
}

switch ($BuildMode) {
    "B" {
        Assert-BuildModeEnv "THYLLORE_FEEDBACK_ENDPOINT" $FeedbackEndpoint
        Assert-BuildModeEnv "THYLLORE_INGEST_TOKEN" $IngestToken
        Assert-BuildModeEnv "THYLLORE_UNLOCK_PUBKEY_B64" $UnlockPubkey
    }
    "C" {
        Assert-BuildModeEnv "THYLLORE_LICENSE_ENDPOINT" $LicenseEndpoint
        Assert-BuildModeEnv "THYLLORE_UNLOCK_PUBKEY_B64" $UnlockPubkey
    }
}

$PlatformConfig = @{
    "win_amd64" = @{
        BlenderName  = "windows-x64"
        WheelSuffix  = "win_amd64"
        # Multiple regex anchors: a wheel is accepted for Windows if any of
        # these matches its filename. This tolerates wheels that come with
        # several platform tags (grpcio dual-tag, maturin manylinux_2_34, ...).
        WheelMatchers = @("win_amd64\.whl$")
    }
    "linux_x86_64" = @{
        BlenderName  = "linux-x64"
        WheelSuffix  = "manylinux2014_x86_64"
        # Accept any glibc-tagged x86_64 Linux wheel:
        #   manylinux2014_x86_64.whl
        #   manylinux_2_17_x86_64.manylinux2014_x86_64.whl (PEP 600 dual)
        #   manylinux_2_34_x86_64.whl (maturin on Ubuntu 24)
        #   linux_x86_64.whl (no manylinux tag — must be vetted manually)
        WheelMatchers = @(
            "manylinux2014_x86_64\.whl$",
            "manylinux_2_\d+_x86_64\.whl$",
            "linux_x86_64\.whl$"
        )
    }
    "macosx_arm64" = @{
        BlenderName  = "macos-arm64"
        WheelSuffix  = "macosx_11_0_arm64"
        # macOS wheels may be arm64-only (cp* native) or universal2
        # (single binary containing both arm64 and x86_64 slices).
        # grpcio and protobuf publish universal2 wheels for cp310, so
        # the arm64-only regex alone drops them and the build fails.
        WheelMatchers = @(
            "macosx_\d+_\d+_arm64\.whl$",
            "macosx_\d+_\d+_universal2\.whl$"
        )
    }
}[$Platform]

Write-Host "[build_blender_addon] Platform: $Platform -> Blender: $($PlatformConfig.BlenderName), wheel: $($PlatformConfig.WheelSuffix), variant: $Variant, build mode: $BuildMode" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Stage directory (mirror blender_addon/ minus excluded paths)
# ---------------------------------------------------------------------------

$StageDir = Join-Path $RepoRoot "build/blender_addon_stage_${Platform}_${Variant}_${BuildMode}"
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

$Source = Join-Path $RepoRoot "blender_addon"

# Tier A files excluded from the lite Variant ZIP. These stay in the source
# tree so Phase 6 (full Variant restoration) does not need to revive them
# from git history; the build script just leaves them out of the artifact.
$LiteExcludeRelPaths = @(
    "operators/auto_rig.py",
    "operators/text_to_mesh.py",
    "operators/_grpc_helpers.py",
    "grpc_client",
    "blender_manifest.toml"
)
# text_to_motion currently still routes through gRPC; exclude it from lite
# until the PyO3 rewrite (phase5_5_onnx/Phase5_5_TextToMotionMlCore.md) ships.
$LiteExcludeRelPaths += "operators/text_to_motion.py"

# Use robocopy on Windows, cp -a on POSIX. We are running PowerShell on
# Windows; if pwsh is invoked on Linux/macOS the path separators still work
# via .NET and we fall back to copy-by-file.
if ($IsWindows -or $env:OS -like "*Windows*") {
    $RoboArgs = @(
        $Source, $StageDir, "/MIR",
        "/XD", "tests", "__pycache__", ".pytest_cache", ".egg-info", "wheels-extracted",
        "/XF", "*.pyc"
    )
    $proc = Start-Process robocopy -ArgumentList $RoboArgs -NoNewWindow -PassThru -Wait
    # robocopy returns 0-7 for success, >=8 for failure
    if ($proc.ExitCode -ge 8) {
        throw "robocopy failed with exit code $($proc.ExitCode)"
    }
} else {
    # POSIX fallback (rsync if available, else cp)
    if (Get-Command rsync -ErrorAction SilentlyContinue) {
        & rsync -a --exclude='tests/' --exclude='__pycache__/' --exclude='wheels-extracted/' --exclude='*.pyc' "$Source/" "$StageDir/"
    } else {
        Copy-Item -Recurse -Force "$Source/*" $StageDir
        Get-ChildItem -Recurse -Force -Directory $StageDir |
            Where-Object { $_.Name -in @("tests", "__pycache__", ".pytest_cache", ".egg-info", "wheels-extracted") } |
            Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Force -File $StageDir -Filter "*.pyc" | Remove-Item -Force
    }
}

# ---------------------------------------------------------------------------
# 1b. Apply Variant-specific pruning (lite removes Tier A files)
# ---------------------------------------------------------------------------

if ($Variant -eq "lite") {
    foreach ($rel in $LiteExcludeRelPaths) {
        $abs = Join-Path $StageDir $rel
        if (Test-Path $abs) {
            Remove-Item -Recurse -Force $abs
            Write-Host "[build_blender_addon] (lite) excluded: $rel" -ForegroundColor DarkYellow
        }
    }
} else {
    # Full Variant uses the canonical manifest, so the lite copy is removed
    # from the stage to avoid shipping a duplicate.
    $LiteManifestStage = Join-Path $StageDir "blender_manifest.lite.toml"
    if (Test-Path $LiteManifestStage) { Remove-Item -Force $LiteManifestStage }
}

$TelemetryStage = Join-Path $StageDir "telemetry"
if ($BuildMode -ne "B" -and (Test-Path $TelemetryStage)) {
    Remove-Item -Recurse -Force $TelemetryStage
    Write-Host "[build_blender_addon] (mode $BuildMode) excluded: telemetry" -ForegroundColor DarkYellow
}
$LicenseClientStage = Join-Path $StageDir "license_client"
if ($BuildMode -ne "C" -and (Test-Path $LicenseClientStage)) {
    Remove-Item -Recurse -Force $LicenseClientStage
    Write-Host "[build_blender_addon] (mode $BuildMode) excluded: license_client" -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------------------
# 2. Filter wheels/ down to platform-matching files (and Variant-allowed names)
# ---------------------------------------------------------------------------

$WheelsDir = Join-Path $StageDir "wheels"
if (-not (Test-Path $WheelsDir)) {
    throw "Stage directory has no wheels/. Run scripts/collect_wheels.ps1 first."
}

# Tier B-only wheel families allowed in the lite Variant. Anything not
# matching is dropped before platform filtering.
$LiteAllowedWheelPrefixes = @("thyllore_ml_core")

$AllWheels = Get-ChildItem -Path $WheelsDir -Filter "*.whl"
$KeptWheels = New-Object System.Collections.Generic.List[string]

foreach ($wheel in $AllWheels) {
    $name = $wheel.Name

    if ($Variant -eq "lite") {
        $isAllowedForLite = $false
        foreach ($prefix in $LiteAllowedWheelPrefixes) {
            if ($name.StartsWith($prefix)) {
                $isAllowedForLite = $true
                break
            }
        }
        if (-not $isAllowedForLite) {
            Remove-Item -Path $wheel.FullName -Force
            continue
        }
    }

    $isPureWheel = $name -match "py3-none-any\.whl$"
    $isPlatformWheel = $false
    foreach ($pattern in $PlatformConfig.WheelMatchers) {
        if ($name -match $pattern) {
            $isPlatformWheel = $true
            break
        }
    }
    if ($isPureWheel -or $isPlatformWheel) {
        $KeptWheels.Add($name)
    } else {
        Remove-Item -Path $wheel.FullName -Force
    }
}

# Also remove HASHES.txt from the staged wheels/ since it is a development
# artifact, not part of the extension.
$StagedHashes = Join-Path $WheelsDir "HASHES.txt"
if (Test-Path $StagedHashes) { Remove-Item -Force $StagedHashes }
$StagedReadme = Join-Path $WheelsDir "README.md"
if (Test-Path $StagedReadme) { Remove-Item -Force $StagedReadme }

$MinWheels = if ($Variant -eq "lite") { 1 } else { 4 }
if ($KeptWheels.Count -lt $MinWheels) {
    throw "Expected at least $MinWheels wheels for $Platform/$Variant, got $($KeptWheels.Count) (found: $($KeptWheels -join ', '))"
}
Write-Host "[build_blender_addon] Kept $($KeptWheels.Count) wheels for $Platform/$Variant" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 3. Substitute placeholders in the Variant's manifest
# ---------------------------------------------------------------------------

# The lite Variant uses blender_manifest.lite.toml (id = thyllore_animation_lite),
# the full Variant uses the canonical blender_manifest.toml. The non-active
# manifest copy is removed from the stage so the ZIP carries exactly one.
$CanonicalManifestStage = Join-Path $StageDir "blender_manifest.toml"
$LiteManifestStage = Join-Path $StageDir "blender_manifest.lite.toml"

if ($Variant -eq "lite") {
    if (-not (Test-Path $LiteManifestStage)) {
        throw "blender_manifest.lite.toml missing from stage; expected at $LiteManifestStage"
    }
    if (Test-Path $CanonicalManifestStage) { Remove-Item -Force $CanonicalManifestStage }
    Move-Item -Force $LiteManifestStage $CanonicalManifestStage
}

$ManifestPath = $CanonicalManifestStage
$ManifestContent = [System.IO.File]::ReadAllText($ManifestPath, [System.Text.Encoding]::UTF8)
$ManifestContent = $ManifestContent -replace 'PLATFORM_BLENDER_NAME', $PlatformConfig.BlenderName

$WheelLines = ($KeptWheels | ForEach-Object { "    `"./wheels/$_`"," }) -join "`n"
$ManifestContent = $ManifestContent -replace '(?ms)wheels = \[.*?\]', "wheels = [`n$WheelLines`n]"

# Use UTF-8 *without BOM* so Blender's TOML parser does not choke on a leading byte order mark.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ManifestPath, $ManifestContent, $Utf8NoBom)

# ---------------------------------------------------------------------------
# 4. ABI marker handshake — wheel ml-api ABI_MARKER must match addon EXPECTED_ABI_MARKER
# ---------------------------------------------------------------------------

$ExpectedAbi = & python -c "import re, pathlib; m = re.search(r'EXPECTED_ABI_MARKER\s*[:=]\s*int\s*=\s*(\d+)', pathlib.Path(r'$RepoRoot/blender_addon/__init__.py').read_text(encoding='utf-8')); print(m.group(1) if m else 'MISSING')"
if ($ExpectedAbi -eq "MISSING") {
    throw "EXPECTED_ABI_MARKER not found in blender_addon/__init__.py"
}

$ApiMarker = & python -c "import re, pathlib; m = re.search(r'pub const ABI_MARKER:\s*u32\s*=\s*(\d+)', pathlib.Path(r'$RepoRoot/crates/thyllore-ml-api/src/lib.rs').read_text(encoding='utf-8')); print(m.group(1) if m else 'MISSING')"
if ($ApiMarker -eq "MISSING") {
    throw "ABI_MARKER not found in crates/thyllore-ml-api/src/lib.rs"
}

if ($ExpectedAbi.Trim() -ne $ApiMarker.Trim()) {
    throw "ABI marker mismatch: addon EXPECTED_ABI_MARKER=$ExpectedAbi vs ml-api ABI_MARKER=$ApiMarker. Bump both or neither."
}
Write-Host "[build_blender_addon] ABI marker verified: $ApiMarker" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 5. Optional: include the Tier B onnx model
# ---------------------------------------------------------------------------

if ($IncludeOnnxModel) {
    $OnnxAbs = if ([System.IO.Path]::IsPathRooted($OnnxSourcePath)) {
        $OnnxSourcePath
    } else {
        Join-Path $RepoRoot $OnnxSourcePath
    }
    if (-not (Test-Path $OnnxAbs)) {
        throw "ONNX model not found at $OnnxAbs (use -IncludeOnnxModel:`$false to skip, or set -OnnxSourcePath)"
    }
    $ModelsDir = Join-Path $StageDir "models"
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    Copy-Item -Path $OnnxAbs -Destination (Join-Path $ModelsDir "curve_copilot.onnx") -Force
    Write-Host "[build_blender_addon] Bundled ONNX model from $OnnxSourcePath" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 5b. Generate build_config.py (BUILD_MODE single source of truth)
# ---------------------------------------------------------------------------

$BuildConfigLines = @("BUILD_MODE = `"$BuildMode`"")
if ($BuildMode -eq "B") {
    $BuildConfigLines += "FEEDBACK_ENDPOINT = `"$FeedbackEndpoint`""
    $BuildConfigLines += "INGEST_TOKEN = `"$IngestToken`""
}
if ($BuildMode -eq "B" -or $BuildMode -eq "C") {
    $BuildConfigLines += "UNLOCK_PUBKEY = `"$UnlockPubkey`""
}
if ($BuildMode -eq "C") {
    $BuildConfigLines += "LICENSE_ENDPOINT = `"$LicenseEndpoint`""
}
$Utf8NoBomConfig = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $StageDir "build_config.py"),
    (($BuildConfigLines -join "`n") + "`n"),
    $Utf8NoBomConfig
)
Write-Host "[build_blender_addon] Generated build_config.py (mode $BuildMode)" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 6. Optional: validate via Blender CLI
# ---------------------------------------------------------------------------

if (-not $SkipBlenderValidate) {
    $BlenderExe = $null
    $BlenderCandidate = Get-ChildItem "C:\Program Files\Blender Foundation" -Filter "blender.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($BlenderCandidate) {
        $BlenderExe = $BlenderCandidate.FullName
    } elseif (Test-Path "/usr/bin/blender") {
        $BlenderExe = "/usr/bin/blender"
    } elseif (Test-Path "/opt/blender/blender") {
        $BlenderExe = "/opt/blender/blender"
    }

    if ($BlenderExe) {
        Write-Host "[build_blender_addon] Validating with $BlenderExe..." -ForegroundColor Cyan
        & $BlenderExe --command extension validate $StageDir
        if ($LASTEXITCODE -ne 0) {
            throw "Blender extension validate failed"
        }
    } else {
        Write-Host "[build_blender_addon] Blender executable not found - skipping validate" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# 7. Create ZIP
# ---------------------------------------------------------------------------

$AbsOutDir = Join-Path $RepoRoot $OutputDir
New-Item -ItemType Directory -Path $AbsOutDir -Force | Out-Null

$ZipBaseName = if ($Variant -eq "lite") {
    "thyllore_animation_lite"
} else {
    "thyllore_animation_addon"
}
if ($BuildMode -ne "A") {
    $ZipBaseName = "${ZipBaseName}_mode_$($BuildMode.ToLower())"
}
$ZipPath = Join-Path $AbsOutDir "$ZipBaseName-$Version-$Platform.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }

# Compress-Archive on Windows PowerShell 5.1 (.NET Framework) emits ZIP entry
# names with backslash separators, which Linux/macOS unzip treats as flat
# filenames -- breaking the addon's package layout. Build the ZIP manually so
# every entry name uses forward slashes, regardless of the host filename.
Add-Type -AssemblyName "System.IO.Compression"
Add-Type -AssemblyName "System.IO.Compression.FileSystem"
$AbsStageDir = (Resolve-Path $StageDir).Path
$Zip = [System.IO.Compression.ZipFile]::Open(
    $ZipPath, [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    Get-ChildItem -Recurse -File -Path $AbsStageDir | ForEach-Object {
        $RelPath = $_.FullName.Substring($AbsStageDir.Length).TrimStart('\', '/').Replace('\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Zip, $_.FullName, $RelPath, [System.IO.Compression.CompressionLevel]::Optimal
        )
    }
}
finally {
    $Zip.Dispose()
}

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

$Hash = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLower()
$SizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)

Write-Host ""
Write-Host "[build_blender_addon] Created: $ZipPath" -ForegroundColor Green
Write-Host "[build_blender_addon] Size:    $SizeMb MiB"
Write-Host "[build_blender_addon] SHA256:  $Hash"

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$SentinelPath = Join-Path $AbsOutDir ".last_built_zip"
[System.IO.File]::WriteAllText($SentinelPath, $ZipPath, $Utf8NoBom)

if ($env:GITHUB_OUTPUT) {
    $GhOut = "zip_path=$ZipPath`nzip_name=$(Split-Path -Leaf $ZipPath)`n"
    [System.IO.File]::AppendAllText($env:GITHUB_OUTPUT, $GhOut, $Utf8NoBom)
}

Remove-Item -Recurse -Force $StageDir
