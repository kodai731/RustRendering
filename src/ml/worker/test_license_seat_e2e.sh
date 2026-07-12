#!/usr/bin/env bash
set -euo pipefail

# Layer-5 e2e: license seat / copy invalidation (mode C), local and npm-free.
#
#   provision -> seat grant -> seat exhaustion (copied key) -> revocation
#   local workerd (real index.mjs + LicenseSeats DO)  --token-->  wheel verify
#
# Asserts the boundary-tests.md layer-5 matrix:
#   - free seat / registered device_id -> unlock_token issued
#   - seat exhaustion (same license_key, extra device_id) -> refused, no token
#   - revoked license -> refused
#   - issued token -> wheel ctx64; no token / expired token -> ctx32
#
# The Durable Object runs in workerd with in-memory storage, so every run
# starts from a clean seat table (idempotent by construction).

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$WORKER_DIR/../../.." && pwd)"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8789"

INGEST_TOKEN="local-e2e-ingest-token"
ADMIN_TOKEN="local-e2e-admin-token"
export INGEST_TOKEN ADMIN_TOKEN

source "$WORKER_DIR/lib_e2e.sh"

e2e_require_tools
e2e_generate_keypair_if_missing
e2e_build_and_install_wheel
e2e_start_worker

WORKER_URL="http://127.0.0.1:$PORT"
LICENSE_KEY="e2e-license-$(date +%s)"
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

echo "[seat-e2e] 1. Provisioning requires the admin token"
check "provision without admin token -> 401" 401 \
    "$(provision "wrong-token" "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"active\"}")"

echo "[seat-e2e] 2. Provision license (max_seats=2, active)"
check "provision -> 204" 204 \
    "$(provision "$ADMIN_TOKEN" "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"active\"}")"

echo "[seat-e2e] 3. Validation and unknown licenses are refused"
check "malformed json -> 400" 400 "$(refresh_status 'not-json')"
check "missing device_id -> 400" 400 "$(refresh_status "{\"license_key\":\"$LICENSE_KEY\"}")"
unknown_response="$(refresh_body '{"license_key":"never-provisioned","device_id":"device-x"}')"
check "unknown license -> refused" "unknown_license" "$(jq -r '.error' <<<"$unknown_response")"

echo "[seat-e2e] 4. Free seat and registered device get tokens"
token_d1="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}" | jq -r '.unlock_token')"
check "device-1 first seat -> token issued" "true" \
    "$([[ -n "$token_d1" && "$token_d1" != "null" ]] && echo true || echo false)"
token_d1_again="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}" | jq -r '.unlock_token')"
check "device-1 registered re-refresh -> token issued" "true" \
    "$([[ -n "$token_d1_again" && "$token_d1_again" != "null" ]] && echo true || echo false)"
token_d2="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-2\"}" | jq -r '.unlock_token')"
check "device-2 second seat -> token issued" "true" \
    "$([[ -n "$token_d2" && "$token_d2" != "null" ]] && echo true || echo false)"

echo "[seat-e2e] 5. Copied license key exhausts seats and is refused"
copy_response="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-3-copy\"}")"
check "device-3 (copy) -> refused" "seat_exhausted" "$(jq -r '.error' <<<"$copy_response")"
check "device-3 (copy) -> no token in response" "false" \
    "$(jq 'has("unlock_token")' <<<"$copy_response")"
check "device-1 still refreshes after copy attempt" 200 \
    "$(refresh_status "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}")"

echo "[seat-e2e] 6. Revoked license is refused"
check "revoke provision -> 204" 204 \
    "$(provision "$ADMIN_TOKEN" "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"revoked\"}")"
revoked_response="$(refresh_body "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}")"
check "revoked license -> refused" "revoked" "$(jq -r '.error' <<<"$revoked_response")"

echo "[seat-e2e] 7. Wheel integration: token gates ctx (64 / 32)"
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
assert tml.effective_context_length(seat_token) == 64, "seat token must unlock ctx64"
assert tml.effective_context_length(None) == 32, "no token must degrade to ctx32"
assert tml.effective_context_length(expired_token) == 32, "expired token must degrade to ctx32"
print("  PASS  seat token -> ctx64, no token -> ctx32, expired token -> ctx32")
PY

echo "[seat-e2e] 8. Wheel refresh_license: full round-trip via the Rust transport"
LICENSE_KEY_WHEEL="e2e-license-wheel-$(date +%s)"
check "provision wheel-test license -> 204" 204 \
    "$(provision "$ADMIN_TOKEN" "{\"license_key\":\"$LICENSE_KEY_WHEEL\",\"max_seats\":1,\"status\":\"active\"}")"
"$LOCAL_DIR/venv/bin/python" - "$WORKER_URL/v1/license/refresh" "$LICENSE_KEY_WHEEL" <<'PY'
import sys
import thyllore_ml_core as tml

endpoint, license_key = sys.argv[1], sys.argv[2]

granted = tml.refresh_license(endpoint, license_key, "wheel-device-1")
assert granted["ok"] is True, granted
assert tml.effective_context_length(granted["unlock_token"]) == 64, granted

copied = tml.refresh_license(endpoint, license_key, "wheel-device-2-copy")
assert copied["ok"] is False and copied["reason"] == "seat_exhausted", copied
assert copied["discard_cached_token"] is True, copied

unknown = tml.refresh_license(endpoint, "never-provisioned", "wheel-device-1")
assert unknown["ok"] is False and unknown["reason"] == "unknown_license", unknown
print("  PASS  wheel refresh_license: grant -> ctx64, copy -> seat_exhausted, unknown -> refused")
PY

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "[seat-e2e] All layer-5 checks passed."
else
    echo "[seat-e2e] $FAILURES check(s) failed." >&2
    exit 1
fi
