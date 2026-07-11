#!/usr/bin/env bash
set -euo pipefail

# Runs worker/src/index.mjs in a local workerd instance so the exact production
# code path (gzip decode, schema validation, Ed25519 signing) can be tested
# without npm or wrangler. The workerd binary is the standalone GitHub release
# (the npm 'workerd' package merely wraps it); it is downloaded with curl and
# pinned by SHA256, so the only trusted artifact is one Cloudflare binary.
#
# Required environment:
#   INGEST_TOKEN                        bearer token the local worker accepts
#   UNLOCK_PRIVATE_KEY_PKCS8_B64[_FILE] Ed25519 PKCS8 DER base64 signing key;
#                                       must match the public key baked into the
#                                       wheel under test
#
# R2 is not bound locally: workerd's r2Bucket binding needs a simulator service
# (what miniflare/npm provides), so the R2 write path is verified against the
# real -test bucket instead (deploy.sh --env test). The empty-batch token
# handshake and every validation branch here never touch R2.

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8787"

WORKERD_VERSION="v1.20260710.1"
WORKERD_GZ_SHA256="4189eaeed663d09b81b41ea0555a7a9fb2ab92e09adf990a4353b4b306e97338"
WORKERD_BINARY_SHA256="801175112d8efcce94f03bd4fd71dcd47cadad9431dbd69edf604e8ff2560f83"

usage() {
    cat <<EOF
Usage: $0 [--port N]

Serves worker/src/index.mjs locally with workerd (foreground).

Options:
  --port N    HTTP port (default: $PORT)
  -h, --help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
    echo "run_local.sh pins the linux-x86_64 workerd build; on other platforms" >&2
    echo "download the matching workerd-<os>-<arch>.gz from" >&2
    echo "  https://github.com/cloudflare/workerd/releases/tag/$WORKERD_VERSION" >&2
    echo "and set its sha256 pins in this script." >&2
    exit 1
fi

if [[ -z "${INGEST_TOKEN:-}" ]]; then
    echo "INGEST_TOKEN is required" >&2
    exit 2
fi
if [[ -n "${UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE:-}" ]]; then
    UNLOCK_PRIVATE_KEY_PKCS8_B64="$(tr -d '\r\n' <"$UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE")"
fi
if [[ -z "${UNLOCK_PRIVATE_KEY_PKCS8_B64:-}" ]]; then
    echo "provide UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE or UNLOCK_PRIVATE_KEY_PKCS8_B64" >&2
    exit 2
fi
export INGEST_TOKEN UNLOCK_PRIVATE_KEY_PKCS8_B64

ensure_workerd() {
    local binary="$LOCAL_DIR/workerd"
    if [[ -x "$binary" ]] \
        && [[ "$(sha256sum "$binary" | awk '{print $1}')" == "$WORKERD_BINARY_SHA256" ]]; then
        printf '%s' "$binary"
        return
    fi

    mkdir -p "$LOCAL_DIR"
    local url="https://github.com/cloudflare/workerd/releases/download/$WORKERD_VERSION/workerd-linux-64.gz"
    local archive="$LOCAL_DIR/workerd.gz"
    echo "[run_local] Downloading workerd $WORKERD_VERSION..." >&2
    curl -sSL --retry 3 -o "$archive" "$url"

    local gz_sha
    gz_sha="$(sha256sum "$archive" | awk '{print $1}')"
    if [[ "$gz_sha" != "$WORKERD_GZ_SHA256" ]]; then
        echo "[run_local] workerd archive sha256 mismatch:" >&2
        echo "  expected $WORKERD_GZ_SHA256" >&2
        echo "  got      $gz_sha" >&2
        rm -f "$archive"
        exit 1
    fi

    gunzip -kf "$archive"
    chmod +x "$binary"
    rm -f "$archive"

    local bin_sha
    bin_sha="$(sha256sum "$binary" | awk '{print $1}')"
    if [[ "$bin_sha" != "$WORKERD_BINARY_SHA256" ]]; then
        echo "[run_local] workerd binary sha256 mismatch after extract" >&2
        exit 1
    fi
    printf '%s' "$binary"
}

WORKERD_BIN="$(ensure_workerd)"

LOCAL_CAPNP="$WORKER_DIR/.workerd.local.capnp"
sed "s/SOCKET_ADDRESS/127.0.0.1:$PORT/" "$WORKER_DIR/workerd.capnp" >"$LOCAL_CAPNP"
trap 'rm -f "$LOCAL_CAPNP"' EXIT

echo "[run_local] Serving worker on http://127.0.0.1:$PORT (Ctrl-C to stop)"
cd "$WORKER_DIR"
exec "$WORKERD_BIN" serve "$LOCAL_CAPNP"
