[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("win_amd64", "linux_x86_64", "macosx_arm64")]
    [string]$Platform,

    [string]$Version = "0.0.1",

    [string]$OutputDir = "dist",

    [switch]$IncludeOnnxModel,

    [string]$OnnxSourcePath = "ml/model/curve_copilot.onnx",

    [switch]$SkipBlenderValidate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."

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

Write-Host "[build_blender_addon] Platform: $Platform -> Blender: $($PlatformConfig.BlenderName), wheel: $($PlatformConfig.WheelSuffix)" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Stage directory (mirror blender_addon/ minus excluded paths)
# ---------------------------------------------------------------------------

$StageDir = Join-Path $RepoRoot "build/blender_addon_stage_$Platform"
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

$Source = Join-Path $RepoRoot "blender_addon"

# Use robocopy on Windows, cp -a on POSIX. We are running PowerShell on
# Windows; if pwsh is invoked on Linux/macOS the path separators still work
# via .NET and we fall back to copy-by-file.
if ($IsWindows -or $env:OS -like "*Windows*") {
    $RoboArgs = @(
        $Source, $StageDir, "/MIR",
        "/XD", "tests", "__pycache__", ".pytest_cache", ".egg-info",
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
        & rsync -a --exclude='tests/' --exclude='__pycache__/' --exclude='*.pyc' "$Source/" "$StageDir/"
    } else {
        Copy-Item -Recurse -Force "$Source/*" $StageDir
        Get-ChildItem -Recurse -Force -Directory $StageDir |
            Where-Object { $_.Name -in @("tests", "__pycache__", ".pytest_cache", ".egg-info") } |
            Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Force -File $StageDir -Filter "*.pyc" | Remove-Item -Force
    }
}

# ---------------------------------------------------------------------------
# 2. Filter wheels/ down to platform-matching files
# ---------------------------------------------------------------------------

$WheelsDir = Join-Path $StageDir "wheels"
if (-not (Test-Path $WheelsDir)) {
    throw "Stage directory has no wheels/. Run scripts/collect_wheels.ps1 first."
}

$AllWheels = Get-ChildItem -Path $WheelsDir -Filter "*.whl"
$KeptWheels = New-Object System.Collections.Generic.List[string]

foreach ($wheel in $AllWheels) {
    $name = $wheel.Name
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

if ($KeptWheels.Count -lt 4) {
    throw "Expected at least 4 wheels for $Platform, got $($KeptWheels.Count) (found: $($KeptWheels -join ', '))"
}
Write-Host "[build_blender_addon] Kept $($KeptWheels.Count) wheels for $Platform" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 3. Substitute placeholders in blender_manifest.toml
# ---------------------------------------------------------------------------

$ManifestPath = Join-Path $StageDir "blender_manifest.toml"
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
    $OnnxAbs = Join-Path $RepoRoot $OnnxSourcePath
    if (-not (Test-Path $OnnxAbs)) {
        throw "ONNX model not found at $OnnxAbs (use -IncludeOnnxModel:`$false to skip, or set -OnnxSourcePath)"
    }
    $ModelsDir = Join-Path $StageDir "models"
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    Copy-Item -Path $OnnxAbs -Destination (Join-Path $ModelsDir "curve_copilot.onnx") -Force
    Write-Host "[build_blender_addon] Bundled ONNX model from $OnnxSourcePath" -ForegroundColor Cyan
}

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

$ZipPath = Join-Path $AbsOutDir "thyllore_animation_addon-$Version-$Platform.zip"
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

# Cleanup stage
Remove-Item -Recurse -Force $StageDir
