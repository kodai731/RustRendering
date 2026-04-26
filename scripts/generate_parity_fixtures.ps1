param(
    [switch]$Force,
    [string]$FixtureRoot = $null,
    [string]$OnnxRevision = $null
)

$ErrorActionPreference = "Stop"

$OnnxRepo = "kodai731/thyllore-curve-copilot"
$OnnxFilename = "curve_copilot.onnx"
if (-not $OnnxRevision) {
    $OnnxRevision = if ($env:THYLLORE_ONNX_REVISION) { $env:THYLLORE_ONNX_REVISION } else { "main" }
}

$ScriptDir = $PSScriptRoot
$WorkspaceRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if (-not $FixtureRoot) {
    if ($env:THYLLORE_PARITY_FIXTURE_OUTPUT) {
        $FixtureRoot = $env:THYLLORE_PARITY_FIXTURE_OUTPUT
    } else {
        $FixtureRoot = Join-Path $WorkspaceRoot "fixtures\ml_parity"
    }
}
Write-Host "fixture root: $FixtureRoot"

$null = New-Item -Force -ItemType Directory -Path `
    (Join-Path $FixtureRoot "proto"), `
    (Join-Path $FixtureRoot "onnx"), `
    (Join-Path $FixtureRoot "numpy")

$OnnxLocal = Join-Path $FixtureRoot "onnx\$OnnxFilename"
$OnnxUrl = "https://huggingface.co/$OnnxRepo/resolve/$OnnxRevision/$OnnxFilename"

if ($Force -or -not (Test-Path $OnnxLocal)) {
    Write-Host "downloading ONNX: $OnnxUrl -> $OnnxLocal"
    Invoke-WebRequest -Uri $OnnxUrl -OutFile "$OnnxLocal.tmp" -UseBasicParsing
    Move-Item -Force "$OnnxLocal.tmp" $OnnxLocal
} else {
    Write-Host "ONNX already present at $OnnxLocal (use -Force to re-download)"
}

$env:THYLLORE_PARITY_FIXTURE_OUTPUT = $FixtureRoot

Push-Location $WorkspaceRoot
try {
    Write-Host "==> generating curve_copilot input + golden fixtures"
    cargo test -p thyllore-ml-core --test curve_copilot_fixture_generator `
        generate_curve_copilot_input_and_golden_fixtures -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) { throw "curve_copilot fixture generation failed" }

    Write-Host "==> generating gRPC proto fixtures"
    cargo test -p thyllore-grpc-client --features auto-rig,text-to-motion `
        --test grpc_fixture_generator generate_grpc_request_response_fixtures `
        -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) { throw "grpc-client fixture generation failed" }
}
finally {
    Pop-Location
    Remove-Item Env:\THYLLORE_PARITY_FIXTURE_OUTPUT -ErrorAction SilentlyContinue
}

$Commit = (& git -C $WorkspaceRoot rev-parse --short=8 HEAD 2>$null)
if (-not $Commit) { $Commit = "unknown" }
$GeneratedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Host "==> writing manifest.json"

$Fixtures = [ordered]@{}
$ExcludedNames = @("manifest.json", "README.md", ".gitkeep")
Get-ChildItem -Recurse -File -Path $FixtureRoot `
    | Where-Object { $ExcludedNames -notcontains $_.Name } `
    | Sort-Object FullName `
    | ForEach-Object {
        $rel = $_.FullName.Substring($FixtureRoot.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
        $Fixtures[$rel] = [ordered]@{
            sha256     = $hash
            size_bytes = $_.Length
        }
    }

$Manifest = [ordered]@{
    schema_version              = 1
    generated_at                = $GeneratedAt
    generator                   = "scripts/generate_parity_fixtures.ps1"
    thyllore_animation_commit   = $Commit
    onnx_huggingface_revision   = $OnnxRevision
    proto_version               = "v1"
    fixtures                    = $Fixtures
}

$Json = $Manifest | ConvertTo-Json -Depth 10
Set-Content -Path (Join-Path $FixtureRoot "manifest.json") -Value $Json -NoNewline
Add-Content -Path (Join-Path $FixtureRoot "manifest.json") -Value ""

Write-Host ""
Write-Host "fixtures regenerated at $FixtureRoot"
Write-Host "ONNX revision: $OnnxRevision"
