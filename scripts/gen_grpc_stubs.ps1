# Generate Python gRPC stubs from the canonical proto file.
#
# Input  : crates/thyllore-grpc-client/proto/animation_ml.proto
# Output : blender_addon/grpc_client/stubs/{animation_ml_pb2.py, animation_ml_pb2_grpc.py, animation_ml_pb2.pyi}
#
# Usage:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/gen_grpc_stubs.ps1
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/gen_grpc_stubs.ps1 -Force
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/gen_grpc_stubs.ps1 -PythonExe python3.10

[CmdletBinding()]
param(
    [switch]$Force,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProtoRoot = Join-Path $RepoRoot "crates/thyllore-grpc-client/proto"
$ProtoFile = Join-Path $ProtoRoot "animation_ml.proto"
$OutDir    = Join-Path $RepoRoot "blender_addon/grpc_client/stubs"
$VenvDir   = Join-Path $RepoRoot ".venv-grpc-tools"

if (-not (Test-Path $ProtoFile)) {
    Write-Error "proto file not found: $ProtoFile"
    exit 1
}

if (-not (Test-Path $VenvDir) -or $Force) {
    if (Test-Path $VenvDir) { Remove-Item $VenvDir -Recurse -Force }
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "venv creation failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    & "$VenvDir/Scripts/python.exe" -m pip install --upgrade pip
    & "$VenvDir/Scripts/python.exe" -m pip install "grpcio-tools>=1.60,<2" "protobuf>=5.26,<6"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = Join-Path $VenvDir "bin/python"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
Get-ChildItem $OutDir -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match "^animation_ml_pb2(_grpc)?\.(py|pyi)$"
} | Remove-Item -Force

& $VenvPython -m grpc_tools.protoc `
    "--proto_path=$ProtoRoot" `
    "--python_out=$OutDir" `
    "--pyi_out=$OutDir" `
    "--grpc_python_out=$OutDir" `
    "$ProtoFile"

if ($LASTEXITCODE -ne 0) {
    Write-Error "grpc_tools.protoc failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

$GrpcStub = Join-Path $OutDir "animation_ml_pb2_grpc.py"
if (-not (Test-Path $GrpcStub)) {
    Write-Error "expected stub not generated: $GrpcStub"
    exit 1
}

$content = Get-Content $GrpcStub -Raw
$patched = $content -replace `
    "(?m)^import animation_ml_pb2 as animation__ml__pb2\s*$", `
    "from . import animation_ml_pb2 as animation__ml__pb2"

if ($content -eq $patched) {
    Write-Warning "import-patch pattern did not match. grpcio-tools output format may have changed."
} else {
    [System.IO.File]::WriteAllText($GrpcStub, $patched)
}

$InitPy = Join-Path $OutDir "__init__.py"
$InitContent = @"
# Auto-managed by scripts/gen_grpc_stubs.ps1.
# Manual edits will be overwritten on regeneration.
from . import animation_ml_pb2 as pb2
from . import animation_ml_pb2_grpc as pb2_grpc

__all__ = ["pb2", "pb2_grpc"]
"@
[System.IO.File]::WriteAllText($InitPy, $InitContent)

Write-Host "Generated:" -ForegroundColor Green
Get-ChildItem $OutDir -File | ForEach-Object {
    Write-Host "  $($_.FullName)"
}
