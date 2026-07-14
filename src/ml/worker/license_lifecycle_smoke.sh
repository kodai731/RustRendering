#!/usr/bin/env bash
set -euo pipefail

# Smoke test for license_lifecycle.sh — verifies the full seat lifecycle:
#   provision (2 seats) -> acquire 2 tokens -> 3rd fails (seat_exhausted)
#   -> revoke -> further requests fail (revoked)
#
# Usage:
#   ADMIN_TOKEN=... INGEST_TOKEN=... FULL_TOKEN_PRIVATE_KEY_PKCS8_B64_FILE=... \
#       ./license_lifecycle_smoke.sh

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$WORKER_DIR/../../.." && pwd)"
LOCAL_DIR="$WORKER_DIR/.local"
PORT="8792"

INGEST_TOKEN="${INGEST_TOKEN:-local-smoke-ingest-token}"
ADMIN_TOKEN="${ADMIN_TOKEN:-local-smoke-admin-token}"
export INGEST_TOKEN ADMIN_TOKEN

source "$WORKER_DIR/lib_e2e.sh"

e2e_require_tools
e2e_generate_keypair_if_missing
e2e_start_worker

WORKER_URL="http://127.0.0.1:$PORT"
export WORKER_URL

echo "[smoke] Target: $WORKER_URL"

FAILURES=0
LICENSE_KEY="smoke-test-$(date +%s)"

check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS  $label"
    else
        echo "  FAIL  $label (expected '$expected', got '$actual')"
        FAILURES=$((FAILURES + 1))
    fi
}

# Step 1: Provision a license with 2 seats using the script
echo ""
echo "[smoke] Step 1: Provision license with 2 seats"
PROVISION_OUTPUT="$(bash "$WORKER_DIR/license_lifecycle.sh" provision 2 "$LICENSE_KEY")"
echo "$PROVISION_OUTPUT" | grep -q "PASS.*provision -> 204" && \
    check "provision via script -> PASS" "true" "true" || \
    check "provision via script -> PASS" "true" "false"

# Step 2: Acquire first seat token
echo ""
echo "[smoke] Step 2: Acquire first seat token"
TOKEN1="$(curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
    -H "Content-Type: application/json" \
    --data "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}" | jq -r '.full_token')"
check "device-1 seat -> token issued" "true" \
    "$([[ -n "$TOKEN1" && "$TOKEN1" != "null" ]] && echo true || echo false)"

# Step 3: Acquire second seat token
echo ""
echo "[smoke] Step 3: Acquire second seat token"
TOKEN2="$(curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
    -H "Content-Type: application/json" \
    --data "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-2\"}" | jq -r '.full_token')"
check "device-2 seat -> token issued" "true" \
    "$([[ -n "$TOKEN2" && "$TOKEN2" != "null" ]] && echo true || echo false)"

# Step 4: Attempt third seat — expect seat_exhausted
echo ""
echo "[smoke] Step 4: Attempt third seat (expect seat_exhausted)"
RESPONSE3="$(curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
    -H "Content-Type: application/json" \
    --data "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-3\"}")"
ERROR3="$(jq -r '.error' <<<"$RESPONSE3")"
check "device-3 -> seat_exhausted" "seat_exhausted" "$ERROR3"

# Step 5: Revoke the license using the script
echo ""
echo "[smoke] Step 5: Revoke license"
REVOKE_OUTPUT="$(bash "$WORKER_DIR/license_lifecycle.sh" revoke "$LICENSE_KEY")"
echo "$REVOKE_OUTPUT" | grep -q "PASS.*revoke -> 204" && \
    check "revoke via script -> PASS" "true" "true" || \
    check "revoke via script -> PASS" "true" "false"

# Step 6: Verify further token requests fail with 'revoked'
echo ""
echo "[smoke] Step 6: Verify revoked license refuses tokens"
REVOKED_RESPONSE="$(curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
    -H "Content-Type: application/json" \
    --data "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"device-1\"}")"
ERROR_REVOKED="$(jq -r '.error' <<<"$REVOKED_RESPONSE")"
check "revoked license -> refused with 'revoked'" "revoked" "$ERROR_REVOKED"

# Summary
echo ""
if [[ $FAILURES -eq 0 ]]; then
    echo "[smoke] All checks passed (license left revoked)."
else
    echo "[smoke] $FAILURES check(s) failed." >&2
    exit 1
fi
