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
$env:THYLLORE_PRIV_PKCS8_B64_PATH = Join-Path $AbsOutputDir "private_key_pkcs8.b64"
$env:THYLLORE_PUB_KEY_PATH = $AbsPublicKeyDest
$env:THYLLORE_PUB_RAW_B64_PATH = Join-Path $AbsOutputDir "public_key.b64"

$PythonScript = @"
import base64
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
priv_pkcs8_der = priv.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
pub_pem = pub.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
pub_raw = pub.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)

Path(os.environ['THYLLORE_PRIV_KEY_PATH']).write_bytes(priv_pem)
Path(os.environ['THYLLORE_PRIV_PKCS8_B64_PATH']).write_text(base64.b64encode(priv_pkcs8_der).decode() + '\n')
Path(os.environ['THYLLORE_PUB_KEY_PATH']).write_bytes(pub_pem)
Path(os.environ['THYLLORE_PUB_RAW_B64_PATH']).write_text(base64.b64encode(pub_raw).decode() + '\n')
print('[gen_license_keypair] Generated:')
print('  private (PEM):        ' + os.environ['THYLLORE_PRIV_KEY_PATH'] + '  (KEEP SECRET)')
print('  private (PKCS8 b64):  ' + os.environ['THYLLORE_PRIV_PKCS8_B64_PATH'] + '  (KEEP SECRET, Worker secret UNLOCK_PRIVATE_KEY_PKCS8_B64)')
print('  public  (PEM):        ' + os.environ['THYLLORE_PUB_KEY_PATH'])
print('  public  (raw b64):    ' + os.environ['THYLLORE_PUB_RAW_B64_PATH'] + '  (THYLLORE_UNLOCK_PUBKEY_B64 for wheel/addon builds)')
"@

& python -c $PythonScript
if ($LASTEXITCODE -ne 0) { throw "Keypair generation failed" }

Write-Host ""
Write-Host "[!] $PrivKeyPath -- DO NOT COMMIT" -ForegroundColor Yellow
Write-Host "[!] Add 'secrets/' to .gitignore (already covered if scripts have been run before)" -ForegroundColor Yellow
