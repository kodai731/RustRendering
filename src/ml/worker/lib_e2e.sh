#!/usr/bin/env bash
# Shared helpers for the local npm-free e2e scripts (test_local_e2e.sh,
# test_license_seat_e2e.sh). Sourced, not executed. Callers must set:
#   REPO_ROOT, WORKER_DIR, LOCAL_DIR, PORT, INGEST_TOKEN, ADMIN_TOKEN
#
# One Ed25519 keypair is shared between both sides: the wheel is built with the
# public key baked in, the local workerd signs full tokens with the private
# key — the exact chain production relies on.

MATURIN="$REPO_ROOT/.venv-collect-wheels/bin/maturin"
PRIV_PKCS8_FILE="$LOCAL_DIR/e2e_priv_pkcs8.b64"
PUB_RAW_FILE="$LOCAL_DIR/e2e_pub_raw.b64"

e2e_require_tools() {
    for tool in curl jq gzip; do
        command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 1; }
    done
    if [[ ! -x "$MATURIN" ]]; then
        echo "maturin not found at $MATURIN (run scripts/collect_wheels.sh once)" >&2
        exit 1
    fi
}

e2e_generate_keypair_if_missing() {
    if [[ -f "$PRIV_PKCS8_FILE" && -f "$PUB_RAW_FILE" ]]; then
        echo "[e2e] Reusing cached keypair in $LOCAL_DIR"
        return
    fi
    echo "[e2e] Generating Ed25519 keypair..."
    mkdir -p "$LOCAL_DIR"
    python3 - "$PRIV_PKCS8_FILE" "$PUB_RAW_FILE" <<'PY'
import base64
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

priv = Ed25519PrivateKey.generate()
pkcs8 = priv.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
pub_raw = priv.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)
open(sys.argv[1], "w").write(base64.b64encode(pkcs8).decode())
open(sys.argv[2], "w").write(base64.b64encode(pub_raw).decode())
PY
}

e2e_build_and_install_wheel() {
    local pub_b64 wheel
    pub_b64="$(cat "$PUB_RAW_FILE")"

    echo "[e2e] Building wheel with the E2E public key baked in..."
    rm -f "$LOCAL_DIR"/wheels/thyllore_ml_core-*.whl 2>/dev/null || true
    mkdir -p "$LOCAL_DIR/wheels"
    (
        cd "$REPO_ROOT/crates/thyllore-ml-core"
        THYLLORE_FULL_TOKEN_PUBKEY_B64="$pub_b64" \
            "$MATURIN" build --release --features python --out "$LOCAL_DIR/wheels" >/dev/null
    )

    if [[ ! -d "$LOCAL_DIR/venv" ]]; then
        echo "[e2e] Creating test venv..."
        python3 -m venv "$LOCAL_DIR/venv"
        "$LOCAL_DIR/venv/bin/pip" install --quiet numpy
    fi
    wheel="$(ls "$LOCAL_DIR"/wheels/thyllore_ml_core-*.whl | head -n1)"
    "$LOCAL_DIR/venv/bin/pip" install --quiet --force-reinstall --no-deps "$wheel"
}

e2e_start_worker() {
    echo "[e2e] Starting local workerd on port $PORT..."
    FULL_TOKEN_PRIVATE_KEY_PKCS8_B64_FILE="$PRIV_PKCS8_FILE" \
        INGEST_TOKEN="$INGEST_TOKEN" ADMIN_TOKEN="$ADMIN_TOKEN" \
        bash "$WORKER_DIR/run_local.sh" --port "$PORT" >"$LOCAL_DIR/workerd.log" 2>&1 &
    E2E_WORKERD_PID=$!
    trap 'kill "$E2E_WORKERD_PID" 2>/dev/null || true' EXIT

    local url="http://127.0.0.1:$PORT"
    for _ in $(seq 1 40); do
        if curl -sS -o /dev/null "$url/v1/message" 2>/dev/null; then
            return
        fi
        sleep 0.25
    done
    echo "[e2e] workerd did not become ready; see $LOCAL_DIR/workerd.log" >&2
    exit 1
}
