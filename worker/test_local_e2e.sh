#!/usr/bin/env bash
set -euo pipefail

# End-to-end test of the production path, entirely local and npm-free:
#
#   local workerd (real index.mjs)  --Ed25519 unlock_token-->  wheel verify
#
# One Ed25519 keypair is used for both sides: the wheel is built with its public
# key baked in, the local worker signs tokens with its private key. A token the
# local worker issues must then unlock ctx64 in the wheel -- the exact chain a
# mode B addon relies on in production.
#
# Requires: curl, jq, gzip, and maturin (from .venv-collect-wheels) + python3.
# R2 writes are out of scope here (see run_local.sh); they are covered by the
# -test bucket via deploy.sh --env test.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_DIR="$REPO_ROOT/worker"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8788"
MATURIN="$REPO_ROOT/.venv-collect-wheels/bin/maturin"

INGEST_TOKEN="local-e2e-ingest-token"
export INGEST_TOKEN

for tool in curl jq gzip; do
    command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 1; }
done
if [[ ! -x "$MATURIN" ]]; then
    echo "maturin not found at $MATURIN (run scripts/collect_wheels.sh once)" >&2
    exit 1
fi

mkdir -p "$LOCAL_DIR"
PRIV_PKCS8_FILE="$LOCAL_DIR/e2e_priv_pkcs8.b64"
PUB_RAW_FILE="$LOCAL_DIR/e2e_pub_raw.b64"

generate_keypair_if_missing() {
    if [[ -f "$PRIV_PKCS8_FILE" && -f "$PUB_RAW_FILE" ]]; then
        echo "[e2e] Reusing cached keypair in $LOCAL_DIR"
        return
    fi
    echo "[e2e] Generating Ed25519 keypair..."
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

build_and_install_wheel() {
    local pub_b64 wheel
    pub_b64="$(cat "$PUB_RAW_FILE")"

    echo "[e2e] Building wheel with the E2E public key baked in..."
    rm -f "$LOCAL_DIR"/wheels/thyllore_ml_core-*.whl 2>/dev/null || true
    mkdir -p "$LOCAL_DIR/wheels"
    (
        cd "$REPO_ROOT/crates/thyllore-ml-core"
        THYLLORE_UNLOCK_PUBKEY_B64="$pub_b64" \
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

generate_keypair_if_missing
build_and_install_wheel

echo "[e2e] Starting local workerd on port $PORT..."
UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE="$PRIV_PKCS8_FILE" INGEST_TOKEN="$INGEST_TOKEN" \
    bash "$WORKER_DIR/run_local.sh" --port "$PORT" >"$LOCAL_DIR/workerd.log" 2>&1 &
WORKERD_PID=$!
trap 'kill "$WORKERD_PID" 2>/dev/null || true' EXIT

WORKER_URL="http://127.0.0.1:$PORT"
for _ in $(seq 1 40); do
    if curl -sS -o /dev/null "$WORKER_URL/v1/message" 2>/dev/null; then
        break
    fi
    sleep 0.25
done

echo "[e2e] Running endpoint smoke checks (local, R2 paths skipped)..."
THYLLORE_INGEST_TOKEN="$INGEST_TOKEN" bash "$WORKER_DIR/smoke.sh" --skip-r2 "$WORKER_URL"

echo "[e2e] Obtaining an unlock token from the local worker..."
token="$(printf '' | gzip -c | curl -sS -X POST "$WORKER_URL/v1/feedback" \
    -H "Authorization: Bearer $INGEST_TOKEN" \
    -H "Content-Encoding: gzip" \
    -H "X-Schema-Version: curve_copilot_feedback/v0" \
    --data-binary @- | jq -r '.unlock_token')"
if [[ -z "$token" || "$token" == "null" ]]; then
    echo "[e2e] FAIL: local worker did not return an unlock token" >&2
    exit 1
fi

echo "[e2e] Verifying the worker-signed token unlocks ctx64 in the wheel..."
"$LOCAL_DIR/venv/bin/python" - "$token" <<'PY'
import sys
import thyllore_ml_core as tml

token = sys.argv[1]
unlocked = tml.effective_context_length(token)
degraded = tml.effective_context_length("garbage-token")
assert unlocked == 64, f"expected ctx64 for worker token, got {unlocked}"
assert degraded == 32, f"expected ctx32 for garbage token, got {degraded}"
print(f"[e2e] PASS: worker token -> ctx{unlocked}, garbage -> ctx{degraded}")
PY

echo "[e2e] All local same-path checks passed."
