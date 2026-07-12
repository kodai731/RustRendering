#!/usr/bin/env bash
set -euo pipefail

# Smoke-tests the license seat endpoints of a DEPLOYED worker with curl only.
# Provisions a single-use random license, walks the seat lifecycle (grant /
# re-grant / exhaustion / revocation) and leaves the license revoked so the
# remote state stays inert.
#
# Usage:
#   WORKER_URL=https://<name>.<subdomain>.workers.dev \
#   THYLLORE_ADMIN_TOKEN=... ./license_smoke.sh
#
# Or pass the URL as the first argument.

WORKER_URL="${1:-${WORKER_URL:-}}"
if [[ -z "$WORKER_URL" ]]; then
    echo "provide the worker URL as \$1 or WORKER_URL" >&2
    exit 2
fi
WORKER_URL="${WORKER_URL%/}"

if [[ -z "${THYLLORE_ADMIN_TOKEN:-}" ]]; then
    echo "THYLLORE_ADMIN_TOKEN is required" >&2
    exit 2
fi

for tool in curl jq openssl; do
    command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 1; }
done

LICENSE_KEY="license-smoke-$(openssl rand -hex 8)"
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
    curl -sS -o /dev/null -w '%{http_code}' -X POST "$WORKER_URL/v1/license/provision" \
        -H "Authorization: Bearer $THYLLORE_ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{\"license_key\":\"$LICENSE_KEY\",\"max_seats\":2,\"status\":\"$1\"}"
}

refresh() {
    curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
        -H "Content-Type: application/json" \
        --data "{\"license_key\":\"$LICENSE_KEY\",\"device_id\":\"$1\"}"
}

echo "[license-smoke] Target: $WORKER_URL (key: $LICENSE_KEY)"

check "provision without admin token -> 401" 401 \
    "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$WORKER_URL/v1/license/provision" \
        -H "Content-Type: application/json" --data '{}')"
check "provision (2 seats, active) -> 204" 204 "$(provision active)"

token_d1="$(refresh smoke-device-1 | jq -r '.full_token')"
check "device-1 seat -> token issued" "true" \
    "$([[ -n "$token_d1" && "$token_d1" != "null" ]] && echo true || echo false)"
token_d1_again="$(refresh smoke-device-1 | jq -r '.full_token')"
check "device-1 re-refresh -> token issued" "true" \
    "$([[ -n "$token_d1_again" && "$token_d1_again" != "null" ]] && echo true || echo false)"
token_d2="$(refresh smoke-device-2 | jq -r '.full_token')"
check "device-2 seat -> token issued" "true" \
    "$([[ -n "$token_d2" && "$token_d2" != "null" ]] && echo true || echo false)"
check "device-3 (copied key) -> seat_exhausted" "seat_exhausted" \
    "$(refresh smoke-device-3 | jq -r '.error')"

check "revoke -> 204" 204 "$(provision revoked)"
check "revoked license -> refused" "revoked" "$(refresh smoke-device-1 | jq -r '.error')"

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "[license-smoke] All checks passed (license left revoked)."
else
    echo "[license-smoke] $FAILURES check(s) failed." >&2
    exit 1
fi
