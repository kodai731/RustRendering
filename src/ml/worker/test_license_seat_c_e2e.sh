#!/usr/bin/env bash
set -euo pipefail

# Layer-5 e2e: C build ZIP license path verification (mode C), local and npm-free.
#
#   Build C-mode ZIP -> extract wheel -> install in venv -> start workerd
#   -> verify license activation: seat_exhausted, revoked, expired token
#
# This tests the production distribution path where the wheel is bundled
# inside a C build ZIP with license_client and build_config.py.

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$WORKER_DIR/../../.." && pwd)"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8791"  # Use different port to avoid conflict with test_license_seat_e2e.sh

INGEST_TOKEN="local-e2e-ingest-token"
ADMIN_TOKEN="local-e2e-admin-token"
export INGEST_TOKEN ADMIN_TOKEN

source "$WORKER_DIR/lib_e2e.sh"

# Fail-fast: check required tools
e2e_require_tools

# Generate keypair if missing (idempotent)
e2e_generate_keypair_if_missing

# Build the wheel with the e2e public key baked in (compile-time option_env!)
pub_b64="$(cat "$PUB_RAW_FILE")"
echo "[czip-e2e] Building wheel with the E2E public key baked in..."
rm -f "$LOCAL_DIR"/wheels/thyllore_ml_core-*.whl 2>/dev/null || true
mkdir -p "$LOCAL_DIR/wheels"
(
    cd "$REPO_ROOT/crates/thyllore-ml-core"
    THYLLORE_FULL_TOKEN_PUBKEY_B64="$pub_b64" \
        "$MATURIN" build --release --features python --out "$LOCAL_DIR/wheels" >/dev/null
)
E2E_WHEEL="$(ls "$LOCAL_DIR"/wheels/thyllore_ml_core-*.whl | head -n1)"

# build_blender_addon.sh bundles wheels from blender_addon/wheels/, so stage the
# e2e wheel there for the ZIP build and restore the original afterwards (idempotent).
ADDON_WHEELS_DIR="$REPO_ROOT/blender_addon/wheels"
ADDON_WHEELS_BACKUP="$LOCAL_DIR/addon_wheels_backup"
restore_addon_wheels() {
    rm -f "$ADDON_WHEELS_DIR"/thyllore_ml_core-*.whl
    mv "$ADDON_WHEELS_BACKUP"/thyllore_ml_core-*.whl "$ADDON_WHEELS_DIR/" 2>/dev/null || true
    rm -rf "$ADDON_WHEELS_BACKUP"
}
rm -rf "$ADDON_WHEELS_BACKUP"
mkdir -p "$ADDON_WHEELS_BACKUP"
mv "$ADDON_WHEELS_DIR"/thyllore_ml_core-*.whl "$ADDON_WHEELS_BACKUP/" 2>/dev/null || true
cp "$E2E_WHEEL" "$ADDON_WHEELS_DIR/"
trap restore_addon_wheels EXIT

# Build the C-mode ZIP with dummy license/feedback env. --output-dir is joined onto
# REPO_ROOT inside the script, so pass it relative to REPO_ROOT.
echo "[czip-e2e] Building C-mode ZIP with build_blender_addon.sh..."
DIST_REL="${LOCAL_DIR#"$REPO_ROOT"/}/dist"
THYLLORE_FEEDBACK_ENDPOINT="http://127.0.0.1:${PORT}/v1/feedback" \
    THYLLORE_INGEST_TOKEN="$INGEST_TOKEN" \
    THYLLORE_LICENSE_ENDPOINT="http://127.0.0.1:${PORT}/v1/license" \
    THYLLORE_FULL_TOKEN_PUBKEY_B64="$pub_b64" \
    bash "$REPO_ROOT/scripts/build_blender_addon.sh" --build-mode C --variant lite \
    --platform linux_x86_64 --output-dir "$DIST_REL"

trap - EXIT
restore_addon_wheels

# Locate the built ZIP (sentinel file tracks it) and extract the bundled wheel.
ZIP_PATH="$(cat "$LOCAL_DIR/dist/.last_built_zip")"
echo "[czip-e2e] Built ZIP: $ZIP_PATH"

EXTRACT_DIR="$LOCAL_DIR/extracted"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q -o "$ZIP_PATH" -d "$EXTRACT_DIR"

EXTRACTED_WHEEL="$(ls "$EXTRACT_DIR"/wheels/thyllore_ml_core-*.whl | head -n1)"
echo "[czip-e2e] Extracted wheel: $EXTRACTED_WHEEL"

# Install the extracted wheel into an isolated venv, then assert against that venv.
if [[ ! -d "$LOCAL_DIR/venv" ]]; then
    echo "[czip-e2e] Creating test venv..."
    python3 -m venv "$LOCAL_DIR/venv"
    "$LOCAL_DIR/venv/bin/pip" install --quiet numpy
fi
"$LOCAL_DIR/venv/bin/pip" install --quiet --force-reinstall --no-deps "$EXTRACTED_WHEEL"

# Start the workerd instance
e2e_start_worker

WORKER_URL="http://127.0.0.1:$PORT"
LICENSE_KEY="e2e-license-czip-$(date +%s)"
FAILURES=0

check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS  $label"
    else
        echo "  FAIL  $label (expected '$expected', got '$actual')"
        FAILURES=$((FAILURES + 1))
    fi
}

provision() {
    local auth_token="$1" body="$2"
    curl -sS -o /dev/null -w '%{http_code}' -X POST "$WORKER_URL/v1/license/provision" \
        -H "Authorization: Bearer $auth_token" \
        -H "Content-Type: application/json" \
        --data "$body"
}

refresh_status() {
    curl -sS -o /dev/null -w '%{http_code}' -X POST "$WORKER_URL/v1/license/refresh" \
        -H "Content-Type: application/json" \
        --data "$1"
}

refresh_body() {
    curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
        -H "Content-Type: application/json" \
        --data "$1"
}

echo "[czip-e2e] 1. Provision license (max_seats=2, active)"
check "provision -> 204" 204 \
    "$(provision "$ADMIN_TOKEN" "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"active\"}")"

echo "[czip-e2e] 2. Free seat gets token"
token_d1="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}" | jq -r '.full_token')"
check "device-1 first seat -> token issued" "true" \
    "$([[ -n "$token_d1" && "$token_d1" != "null" ]] && echo true || echo false)"

echo "[czip-e2e] 3. Second seat gets token"
token_d2="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-2\"}" | jq -r '.full_token')"
check "device-2 second seat -> token issued" "true" \
    "$([[ -n "$token_d2" && "$token_d2" != "null" ]] && echo true || echo false)"

echo "[czip-e2e] 4. Seat exhaustion (copy attempt)"
copy_response="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-3-copy\"}")"
check "device-3 (copy) -> refused" "seat_exhausted" "$(jq -r '.error' <<<"$copy_response")"

echo "[czip-e2e] 5. Revoked license is refused"
check "revoke provision -> 204" 204 \
    "$(provision "$ADMIN_TOKEN" "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"revoked\"}")"
revoked_response="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}")"
check "revoked license -> refused" "revoked" "$(jq -r '.error' <<<"$revoked_response")"

echo "[czip-e2e] 6. Expired token degrades to ctx32"
expired_token="$(python3 - "$PRIV_PKCS8_FILE" <<'PY'
import base64
import json
import sys
import time
from cryptography.hazmat.primitives.serialization import load_der_private_key

der = base64.b64decode(open(sys.argv[1]).read().strip())
key = load_der_private_key(der, password=None)
payload = json.dumps({"exp": int(time.time()) - 60}).encode()
b64url = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
print(f"{b64url(payload)}.{b64url(key.sign(payload))}")
PY
)"

"$LOCAL_DIR/venv/bin/python" - "$token_d1" "$expired_token" <<'PY'
import sys
import thyllore_ml_core as tml

seat_token, expired_token = sys.argv[1], sys.argv[2]
assert tml.effective_context_length(seat_token) == 64, "seat token must keep ctx64"
assert tml.effective_context_length(None) == 32, "no token must degrade to ctx32"
assert tml.effective_context_length(expired_token) == 32, "expired token must degrade to ctx32"
print("  PASS  seat token -> ctx64, no token -> ctx32, expired token -> ctx32")
PY

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "[czip-e2e] All C build ZIP license path checks passed."
else
    echo "[czip-e2e] $FAILURES check(s) failed." >&2
    exit 1
fi
