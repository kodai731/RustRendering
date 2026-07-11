#!/usr/bin/env bash
set -euo pipefail

# Smoke-tests a deployed Curve Copilot feedback Worker with curl only (no npm,
# no workerd). Exercises the live endpoints so no local Workers runtime is
# needed to verify behaviour before baking the URL into mode B builds.
#
# Usage:
#   WORKER_URL=https://<name>.<subdomain>.workers.dev \
#   THYLLORE_INGEST_TOKEN=... ./smoke.sh
#
# Or pass the URL as the first argument.

SKIP_R2=0
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-r2) SKIP_R2=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

WORKER_URL="${1:-${WORKER_URL:-}}"
if [[ -z "$WORKER_URL" ]]; then
    echo "provide the worker URL as \$1 or WORKER_URL" >&2
    exit 2
fi
WORKER_URL="${WORKER_URL%/}"

if [[ -z "${THYLLORE_INGEST_TOKEN:-}" ]]; then
    echo "THYLLORE_INGEST_TOKEN is required" >&2
    exit 2
fi

for tool in curl jq gzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "required tool not found: $tool" >&2
        exit 1
    fi
done

FAILURES=0

status_of() {
    curl -sS -o /dev/null -w '%{http_code}' "$@"
}

check_status() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS  $label (HTTP $actual)"
    else
        echo "  FAIL  $label (expected $expected, got $actual)"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "[smoke] Target: $WORKER_URL"

echo "[smoke] 1. Unauthorized message is rejected"
unauth_status="$(status_of -X POST "$WORKER_URL/v1/message" \
    -H "Content-Type: application/json" \
    --data '{"text":"unauthorized probe"}')"
check_status "POST /v1/message without token" 401 "$unauth_status"

if [[ $SKIP_R2 -eq 1 ]]; then
    echo "[smoke] 2. Authorized message (skipped: R2 write path, use the -test bucket)"
else
    echo "[smoke] 2. Authorized message is accepted"
    message_status="$(status_of -X POST "$WORKER_URL/v1/message" \
        -H "Authorization: Bearer $THYLLORE_INGEST_TOKEN" \
        -H "Content-Type: application/json" \
        --data '{"text":"smoke test","addon_version":"0.0.1","anon_id":"smoke","ts":0}')"
    check_status "POST /v1/message with token" 204 "$message_status"
fi

echo "[smoke] 3. Empty feedback batch returns an unlock token"
feedback_response="$(printf '' | gzip -c | curl -sS -X POST "$WORKER_URL/v1/feedback" \
    -H "Authorization: Bearer $THYLLORE_INGEST_TOKEN" \
    -H "Content-Encoding: gzip" \
    -H "X-Schema-Version: curve_copilot_feedback/v0" \
    --data-binary @-)"
if jq -e 'has("unlock_token") and has("exp")' >/dev/null 2>&1 <<<"$feedback_response"; then
    exp="$(jq -r '.exp' <<<"$feedback_response")"
    echo "  PASS  POST /v1/feedback returns unlock_token (exp=$exp)"
else
    echo "  FAIL  POST /v1/feedback did not return an unlock token"
    echo "        response: $feedback_response"
    FAILURES=$((FAILURES + 1))
fi

echo
if [[ $FAILURES -eq 0 ]]; then
    echo "[smoke] All checks passed."
else
    echo "[smoke] $FAILURES check(s) failed." >&2
    exit 1
fi
