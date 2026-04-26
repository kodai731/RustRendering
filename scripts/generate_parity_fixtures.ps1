param(
    [switch]$Force,
    [string]$SharedDataPath = $null
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$WorkspaceRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PathsFile = Join-Path $WorkspaceRoot ".claude/local/paths.md"

if (-not $SharedDataPath) {
    if (-not (Test-Path $PathsFile)) {
        throw "paths.md not found at $PathsFile"
    }
    $sharedDataMatches = Select-String -Path $PathsFile -Pattern "^-\s*SharedDataPath\s*=\s*(.+)$"
    if (-not $sharedDataMatches) {
        throw "SharedDataPath not found in $PathsFile"
    }
    $SharedDataPath = $sharedDataMatches[0].Matches[0].Groups[1].Value.Trim()
}

# `wsl.localhost` UNC must use forward slashes for std::fs on Windows.
$SharedDataPathForRust = $SharedDataPath
if ($SharedDataPathForRust.StartsWith("\\")) {
    $SharedDataPathForRust = $SharedDataPathForRust.Replace("\", "/")
}

$FixtureRoot = "$SharedDataPath\fixtures\ml_parity"
$FixtureRootForRust = "$SharedDataPathForRust/fixtures/ml_parity"
Write-Host "fixture root (Windows view): $FixtureRoot"
Write-Host "fixture root (Rust view):    $FixtureRootForRust"

$null = New-Item -Force -ItemType Directory -Path "$FixtureRoot\glb" `
    , "$FixtureRoot\proto", "$FixtureRoot\onnx", "$FixtureRoot\numpy"

$ExportsDir = "$SharedDataPath\exports"
if (-not (Test-Path $ExportsDir)) {
    throw "Exports dir not found: $ExportsDir"
}

$LatestOnnx = Get-ChildItem -Path $ExportsDir -Filter "curve_copilot_*.onnx" `
    | Sort-Object Name `
    | Select-Object -Last 1
if (-not $LatestOnnx) {
    throw "No curve_copilot_*.onnx found in $ExportsDir"
}
Write-Host "copying onnx: $($LatestOnnx.FullName) -> $FixtureRoot\onnx\curve_copilot.onnx"
Copy-Item -Force $LatestOnnx.FullName "$FixtureRoot\onnx\curve_copilot.onnx"

$env:THYLLORE_PARITY_FIXTURE_OUTPUT = $FixtureRootForRust

Push-Location $WorkspaceRoot
try {
    Write-Host "==> generating Tier B (curve_copilot) input + golden fixtures"
    cargo test -p thyllore-ml-core --test curve_copilot_fixture_generator `
        generate_curve_copilot_input_and_golden_fixtures -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) { throw "ml-core fixture generation failed" }

    Write-Host "==> generating Tier A proto fixtures"
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
    schema_version            = 1
    generated_at              = $GeneratedAt
    generator                 = "scripts/generate_parity_fixtures.ps1"
    thyllore_animation_commit = $Commit
    proto_version             = "v1"
    fixtures                  = $Fixtures
}

$Json = $Manifest | ConvertTo-Json -Depth 10
Set-Content -Path "$FixtureRoot\manifest.json" -Value $Json -NoNewline
Add-Content -Path "$FixtureRoot\manifest.json" -Value ""

Write-Host ""
Write-Host "fixtures regenerated at $FixtureRoot"
Write-Host "next: commit manifest.json + new files in SharedData (run from WSL2 bash recommended):"
Write-Host "  wsl -d Ubuntu -- bash -lc 'cd /home/kodai/Projects/SharedData && git add fixtures/ml_parity && git commit -m `"regenerate ml_parity fixtures`"'"
