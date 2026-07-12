#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR="${1:-secrets}"
PUBLIC_KEY_DEST="${2:-secrets/public_key.pem}"

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey" 2>/dev/null; then
    echo "[gen_license_keypair] python package 'cryptography' is required:" >&2
    echo "  $PYTHON_BIN -m pip install 'cryptography>=42'" >&2
    exit 1
fi

ABS_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
ABS_PUBLIC_KEY_DEST="$REPO_ROOT/$PUBLIC_KEY_DEST"
mkdir -p "$ABS_OUTPUT_DIR" "$(dirname "$ABS_PUBLIC_KEY_DEST")"

THYLLORE_PRIV_KEY_PATH="$ABS_OUTPUT_DIR/private_key.pem" \
THYLLORE_PRIV_PKCS8_B64_PATH="$ABS_OUTPUT_DIR/private_key_pkcs8.b64" \
THYLLORE_PUB_KEY_PATH="$ABS_PUBLIC_KEY_DEST" \
THYLLORE_PUB_RAW_B64_PATH="$ABS_OUTPUT_DIR/public_key.b64" \
"$PYTHON_BIN" - <<'PY'
import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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

Path(os.environ["THYLLORE_PRIV_KEY_PATH"]).write_bytes(priv_pem)
Path(os.environ["THYLLORE_PRIV_PKCS8_B64_PATH"]).write_text(
    base64.b64encode(priv_pkcs8_der).decode() + "\n"
)
Path(os.environ["THYLLORE_PUB_KEY_PATH"]).write_bytes(pub_pem)
Path(os.environ["THYLLORE_PUB_RAW_B64_PATH"]).write_text(
    base64.b64encode(pub_raw).decode() + "\n"
)

print("[gen_license_keypair] Generated:")
print("  private (PEM):        " + os.environ["THYLLORE_PRIV_KEY_PATH"] + "  (KEEP SECRET)")
print("  private (PKCS8 b64):  " + os.environ["THYLLORE_PRIV_PKCS8_B64_PATH"] + "  (KEEP SECRET, Worker secret FULL_TOKEN_PRIVATE_KEY_PKCS8_B64)")
print("  public  (PEM):        " + os.environ["THYLLORE_PUB_KEY_PATH"])
print("  public  (raw b64):    " + os.environ["THYLLORE_PUB_RAW_B64_PATH"] + "  (THYLLORE_FULL_TOKEN_PUBKEY_B64 for wheel/addon builds)")
PY

echo
echo "[!] $ABS_OUTPUT_DIR/private_key.pem and private_key_pkcs8.b64 -- DO NOT COMMIT (secrets/ is gitignored)"
