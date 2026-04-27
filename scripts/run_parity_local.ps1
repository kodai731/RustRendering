[CmdletBinding()]
param(
    [string]$BlenderExe,
    [switch]$SkipFixtureGeneration,
    [switch]$SkipBuild,
    [switch]$ForceOnnxRuntime,
    [switch]$ForceBlenderDownload,
    [ValidateSet("all", "blender_parity", "operator_smoke", "wire_parity")]
    [string]$TestSubset = "all",
    [string]$OnnxRevision = "main"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $RepoRoot

Add-Type -AssemblyName "System.IO.Compression.FileSystem"

$BlenderVersion = "4.2.0"
$BlenderArchiveUrl = "https://download.blender.org/release/Blender4.2/blender-${BlenderVersion}-windows-x64.zip"
if ($env:THYLLORE_BLENDER_INSTALL_DIR) {
    $BlenderInstallDir = $env:THYLLORE_BLENDER_INSTALL_DIR
} else {
    $BlenderInstallDir = Join-Path $env:LOCALAPPDATA "Thyllore/blender-${BlenderVersion}"
}
$BlenderExtractedDir = Join-Path $BlenderInstallDir "blender-${BlenderVersion}-windows-x64"
$DefaultBlenderExe = Join-Path $BlenderExtractedDir "blender.exe"

$OrtVersion = "1.23.2"
$OrtVendorDir = Join-Path $RepoRoot "vendor/onnxruntime/onnxruntime-win-x64-${OrtVersion}"
$OrtDylibPath = Join-Path $OrtVendorDir "lib/onnxruntime.dll"
$OrtArchiveUrl = "https://github.com/microsoft/onnxruntime/releases/download/v${OrtVersion}/onnxruntime-win-x64-${OrtVersion}.zip"

$FixtureRoot = Join-Path $RepoRoot "fixtures/ml_parity"
$OnnxRepo = "kodai731/thyllore-curve-copilot"
$OnnxLocalPath = Join-Path $FixtureRoot "onnx/curve_copilot.onnx"
$OnnxDownloadUrl = "https://huggingface.co/${OnnxRepo}/resolve/${OnnxRevision}/curve_copilot.onnx"

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Invoke-Download {
    param([string]$Url, [string]$Destination)
    Write-Host "  curl -> $Destination"
    & curl.exe -L --fail --retry 3 --retry-delay 2 -o $Destination $Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $Url" }
}

function Expand-ZipFile {
    param([string]$Archive, [string]$Destination)
    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Destination)
}

function Resolve-PowerShellExe {
    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($pwsh) { return $pwsh.Source }
    return (Get-Command powershell).Source
}

function Resolve-BlenderExecutable {
    if ($BlenderExe -and (Test-Path $BlenderExe)) {
        return (Resolve-Path $BlenderExe).Path
    }
    if ($env:THYLLORE_BLENDER_PATH -and (Test-Path $env:THYLLORE_BLENDER_PATH)) {
        return (Resolve-Path $env:THYLLORE_BLENDER_PATH).Path
    }
    $systemBlender = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
    if (Test-Path $systemBlender) { return $systemBlender }
    if (Test-Path $DefaultBlenderExe) { return $DefaultBlenderExe }
    return $null
}

function Install-Blender42 {
    Write-Step "Downloading Blender ${BlenderVersion} to $BlenderInstallDir"
    New-Item -ItemType Directory -Path $BlenderInstallDir -Force | Out-Null
    $archive = Join-Path $BlenderInstallDir "blender.zip"
    Invoke-Download -Url $BlenderArchiveUrl -Destination $archive
    Write-Host "  extracting..."
    Expand-ZipFile -Archive $archive -Destination $BlenderInstallDir
    Remove-Item -Force $archive
}

function Update-BlenderBundledCRT {
    param([string]$BlenderRoot)
    $blenderCrtDir = Join-Path $BlenderRoot "blender.crt"
    if (-not (Test-Path $blenderCrtDir)) { return }

    $bundledMsvcp = Join-Path $blenderCrtDir "msvcp140.dll"
    $systemMsvcp = "C:\Windows\System32\msvcp140.dll"
    if (-not (Test-Path $systemMsvcp)) { return }

    $bundledVer = [Version]((Get-Item $bundledMsvcp).VersionInfo.ProductVersion)
    $systemVer = [Version]((Get-Item $systemMsvcp).VersionInfo.ProductVersion)
    if ($bundledVer -ge $systemVer) {
        Write-Host "Blender bundled CRT ($bundledVer) >= system ($systemVer); no upgrade needed"
        return
    }

    Write-Step "Upgrading Blender bundled CRT $bundledVer -> $systemVer (onnxruntime 1.23.2 needs newer MSVC ABI)"
    $crtNames = @(
        "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
        "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"
    )
    foreach ($name in $crtNames) {
        $src = Join-Path "C:\Windows\System32" $name
        if (Test-Path $src) {
            Copy-Item -Force $src (Join-Path $blenderCrtDir $name)
        }
    }
}

function Install-OnnxRuntime {
    Write-Step "Downloading ONNX Runtime ${OrtVersion} to vendor/onnxruntime/"
    $vendorRoot = Join-Path $RepoRoot "vendor/onnxruntime"
    New-Item -ItemType Directory -Path $vendorRoot -Force | Out-Null
    $archive = Join-Path $vendorRoot "onnxruntime-win-x64-${OrtVersion}.zip"
    Invoke-Download -Url $OrtArchiveUrl -Destination $archive
    Write-Host "  extracting..."
    Expand-ZipFile -Archive $archive -Destination $vendorRoot
    Remove-Item -Force $archive
}

function Download-CurveCopilotOnnx {
    Write-Step "Downloading curve_copilot.onnx from HuggingFace ($OnnxRepo @ $OnnxRevision)"
    $onnxDir = Split-Path -Parent $OnnxLocalPath
    New-Item -ItemType Directory -Path $onnxDir -Force | Out-Null
    Invoke-Download -Url $OnnxDownloadUrl -Destination $OnnxLocalPath
}

function Invoke-CargoTest {
    param(
        [Parameter(Mandatory = $true)][string]$Crate,
        [Parameter(Mandatory = $true)][string]$TestFile,
        [string]$TestName,
        [string[]]$Features,
        [switch]$Ignored
    )
    $cargoArgs = @("test", "-p", $Crate, "--test", $TestFile)
    if ($Features) {
        $cargoArgs += @("--features", ($Features -join ","))
    }
    if ($TestName) {
        $cargoArgs += $TestName
    }
    $cargoArgs += @("--", "--nocapture")
    if ($Ignored) {
        $cargoArgs += "--ignored"
    }
    Write-Host "  cargo $($cargoArgs -join ' ')"
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "cargo test failed: $Crate / $TestFile"
    }
}

Write-Step "Resolving prerequisites"

if ($ForceOnnxRuntime -or -not (Test-Path $OrtDylibPath)) {
    Install-OnnxRuntime
}
Write-Host "ONNX Runtime: $OrtDylibPath"

$resolvedBlender = Resolve-BlenderExecutable
if (-not $resolvedBlender -or $ForceBlenderDownload) {
    Install-Blender42
    $resolvedBlender = $DefaultBlenderExe
}
if (-not (Test-Path $resolvedBlender)) {
    throw "Blender executable not found at $resolvedBlender"
}
Write-Host "Blender: $resolvedBlender"
& $resolvedBlender --version | Select-Object -First 1

Update-BlenderBundledCRT -BlenderRoot (Split-Path -Parent $resolvedBlender)

if (-not (Test-Path $OnnxLocalPath)) {
    Download-CurveCopilotOnnx
}
Write-Host "curve_copilot.onnx: $OnnxLocalPath"

$psExe = Resolve-PowerShellExe

$publicKeyPath = Join-Path $RepoRoot "blender_addon/license/public_key.pem"
$publicKeyBackup = $null
if (Test-Path $publicKeyPath) {
    $publicKeyBackup = [System.IO.File]::ReadAllBytes($publicKeyPath)
}

function Restore-PublicKey {
    if ($script:publicKeyBackup -and (Test-Path $script:publicKeyPath)) {
        $current = [System.IO.File]::ReadAllBytes($script:publicKeyPath)
        if (-not [System.Linq.Enumerable]::SequenceEqual([byte[]]$current, [byte[]]$script:publicKeyBackup)) {
            [System.IO.File]::WriteAllBytes($script:publicKeyPath, $script:publicKeyBackup)
            Write-Host "Restored $script:publicKeyPath to original"
        }
    }
}

trap {
    Restore-PublicKey
    break
}

if (-not $SkipBuild) {
    Write-Step "Generating dev keypair (will be restored after run)"
    & $psExe -NoProfile -ExecutionPolicy Bypass -File "$RepoRoot/scripts/gen_license_keypair.ps1"
    if ($LASTEXITCODE -ne 0) { throw "gen_license_keypair.ps1 failed" }

    Write-Step "Collecting vendored wheels (win_amd64)"
    & $psExe -NoProfile -ExecutionPolicy Bypass -File "$RepoRoot/scripts/collect_wheels.ps1" `
        -Platforms "win_amd64"
    if ($LASTEXITCODE -ne 0) { throw "collect_wheels.ps1 failed" }

    Write-Step "Building Blender extension ZIP (win_amd64)"
    & $psExe -NoProfile -ExecutionPolicy Bypass -File "$RepoRoot/scripts/build_blender_addon.ps1" `
        -Platform "win_amd64" -SkipBlenderValidate
    if ($LASTEXITCODE -ne 0) { throw "build_blender_addon.ps1 failed" }
}

Write-Step "Installing Blender extension into user dir"
$zip = Get-ChildItem (Join-Path $RepoRoot "dist") -Filter "thyllore_animation_addon-*-win_amd64.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $zip) {
    throw "No extension ZIP found under dist/. Re-run without -SkipBuild."
}

$existingExtensionDir = Join-Path $env:APPDATA "Blender Foundation\Blender\4.2\extensions\user_default\thyllore_animation"
if (Test-Path $existingExtensionDir) {
    Write-Host "Removing existing extension at $existingExtensionDir"
    Remove-Item -Recurse -Force $existingExtensionDir
}

Write-Host "Installing $($zip.FullName)"
& $resolvedBlender --command extension install-file --repo user_default --enable $zip.FullName
if ($LASTEXITCODE -ne 0) { throw "extension install-file failed" }

$env:THYLLORE_PARITY_FIXTURE_OUTPUT = $FixtureRoot
$env:THYLLORE_BLENDER_PATH = $resolvedBlender
$env:THYLLORE_TEST_BYPASS_LICENSE = "1"
$env:THYLLORE_HEADLESS = "1"
$env:ORT_DYLIB_PATH = $OrtDylibPath

if (-not $SkipFixtureGeneration) {
    Write-Step "Generating proto/numpy parity fixtures"
    Invoke-CargoTest -Crate "thyllore-grpc-client" -TestFile "grpc_fixture_generator" `
        -TestName "generate_grpc_request_response_fixtures" `
        -Features @("auto-rig", "text-to-motion") -Ignored
    Invoke-CargoTest -Crate "thyllore-ml-core" -TestFile "curve_copilot_fixture_generator" `
        -TestName "generate_curve_copilot_input_and_golden_fixtures" -Ignored
}

$runBlenderParity = ($TestSubset -in @("all", "blender_parity"))
$runOperatorSmoke = ($TestSubset -in @("all", "operator_smoke"))
$runWireParity = ($TestSubset -in @("all", "wire_parity"))

if ($runBlenderParity) {
    Write-Step "Running curve_copilot_blender_parity (typed vs call_op_json)"
    Invoke-CargoTest -Crate "thyllore-ml-core" -TestFile "curve_copilot_blender_parity" `
        -TestName "curve_copilot_typed_and_call_op_paths_match_in_blender" -Ignored
}

if ($runOperatorSmoke) {
    Write-Step "Running curve_copilot_operator_smoke"
    Invoke-CargoTest -Crate "thyllore-grpc-client" -TestFile "curve_copilot_operator_smoke" `
        -TestName "curve_copilot_operator_runs_under_background" `
        -Features @("auto-rig", "text-to-motion") -Ignored
}

if ($runWireParity) {
    Write-Step "Running grpc_request_wire_parity"
    Invoke-CargoTest -Crate "thyllore-grpc-client" -TestFile "grpc_request_wire_parity" `
        -TestName "grpc_request_wire_bytes_match_across_rust_and_blender" `
        -Features @("auto-rig", "text-to-motion") -Ignored
}

Restore-PublicKey

Write-Step "All requested parity tests passed locally"
