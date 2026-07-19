#!/usr/bin/env bash
set -euo pipefail

# End-to-end test of the production mode B path, entirely local and npm-free:
#
#   local workerd (real index.mjs)  --Ed25519 full_token-->  wheel verify
#
# Requires: curl, jq, gzip, and maturin (from .venv-collect-wheels) + python3.
# R2 writes are out of scope here (see run_local.sh); they are covered by the
# -test bucket via deploy.sh --env test. Mode C (private) never contacts the
# worker, so there is no mode C e2e.

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$WORKER_DIR/../../.." && pwd)"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8788"

INGEST_TOKEN="local-e2e-ingest-token"
export INGEST_TOKEN

source "$WORKER_DIR/lib_e2e.sh"

e2e_require_tools
e2e_generate_keypair_if_missing
e2e_build_and_install_wheel
e2e_start_worker

WORKER_URL="http://127.0.0.1:$PORT"

echo "[e2e] Running endpoint smoke checks (local, R2 paths skipped)..."
THYLLORE_INGEST_TOKEN="$INGEST_TOKEN" bash "$WORKER_DIR/smoke.sh" --skip-r2 "$WORKER_URL"

echo "[e2e] Obtaining a full token from the local worker..."
token="$(printf '' | gzip -c | curl -sS -X POST "$WORKER_URL/v1/feedback" \
    -H "Authorization: Bearer $INGEST_TOKEN" \
    -H "Content-Encoding: gzip" \
    -H "X-Schema-Version: curve_copilot_feedback/v1" \
    --data-binary @- | jq -r '.full_token')"
if [[ -z "$token" || "$token" == "null" ]]; then
    echo "[e2e] FAIL: local worker did not return a full token" >&2
    exit 1
fi

echo "[e2e] Verifying the worker-signed token keeps ctx64 in the wheel..."
"$LOCAL_DIR/venv/bin/python" - "$token" <<'PY'
import sys
import thyllore_ml_core as tml

token = sys.argv[1]
full_ctx = tml.effective_context_length(token)
degraded = tml.effective_context_length("garbage-token")
assert full_ctx == 64, f"expected ctx64 for worker token, got {full_ctx}"
assert degraded == 32, f"expected ctx32 for garbage token, got {degraded}"
print(f"[e2e] PASS: worker token -> ctx{full_ctx}, garbage -> ctx{degraded}")
PY

echo "[e2e] All local same-path checks passed."
