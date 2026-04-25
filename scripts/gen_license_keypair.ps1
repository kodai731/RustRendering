[CmdletBinding()]
param(
    [string]$OutputDir = "secrets",
    [string]$PublicKeyDest = "blender_addon/license/public_key.pem"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."

# Ensure cryptography is available in the active Python.
$cryptoCheck = & python -c "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[gen_license_keypair] Installing cryptography..."
    & python -m pip install --quiet "cryptography>=42"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install cryptography" }
}

$AbsOutputDir = Join-Path $RepoRoot $OutputDir
$AbsPublicKeyDest = Join-Path $RepoRoot $PublicKeyDest

New-Item -ItemType Directory -Path $AbsOutputDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $AbsPublicKeyDest) -Force | Out-Null

$PrivKeyPath = Join-Path $AbsOutputDir "private_key.pem"

$env:THYLLORE_PRIV_KEY_PATH = $PrivKeyPath
$env:THYLLORE_PUB_KEY_PATH = $AbsPublicKeyDest

$PythonScript = @"
import os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

priv = Ed25519PrivateKey.generate()
pub = priv.public_key()

priv_pem = priv.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
pub_pem = pub.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

priv_path = os.environ['THYLLORE_PRIV_KEY_PATH']
pub_path = os.environ['THYLLORE_PUB_KEY_PATH']
Path(priv_path).write_bytes(priv_pem)
Path(pub_path).write_bytes(pub_pem)
print('[gen_license_keypair] Generated:')
print('  private: ' + priv_path + '  (KEEP SECRET)')
print('  public:  ' + pub_path)
"@

& python -c $PythonScript
if ($LASTEXITCODE -ne 0) { throw "Keypair generation failed" }

Write-Host ""
Write-Host "[!] $PrivKeyPath -- DO NOT COMMIT" -ForegroundColor Yellow
Write-Host "[!] Add 'secrets/' to .gitignore (already covered if scripts have been run before)" -ForegroundColor Yellow
