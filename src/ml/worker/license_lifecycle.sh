#!/usr/bin/env bash
set -euo pipefail

# Reusable CLI for license lifecycle management against a local workerd instance.
#
# Usage:
#   ./license_lifecycle.sh provision <max_seats>
#   ./license_lifecycle.sh revoke <license_id>
#   ./license_lifecycle.sh status <license_id>
#
# Environment (same as run_local.sh):
#   ADMIN_TOKEN          bearer token for /v1/license/provision
#   WORKER_URL           base URL of the workerd instance (default: http://127.0.0.1:8787)
#
# Each operation prints a PASS/FAIL line so it can be used in test scripts.

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_URL="${WORKER_URL:-http://127.0.0.1:8787}"
WORKER_URL="${WORKER_URL%/}"

if [[ -z "${ADMIN_TOKEN:-}" ]]; then
    echo "ADMIN_TOKEN is required" >&2
    exit 2
fi

for tool in curl jq; do
    command -v "$tool" >/dev/null 2>&1 || { echo "required tool not found: $tool" >&2; exit 1; }
done

usage() {
    cat <<EOF
Usage: $0 <command> [args...]

Commands:
  provision <max_seats>              Provision a new license with the given max seats (status=active)
  revoke <license_id>                Revoke an existing license
  status <license_id>                Query the current status of a license

Environment:
  ADMIN_TOKEN    Bearer token for admin operations (required)
  WORKER_URL     Base URL of workerd (default: http://127.0.0.1:8787)
EOF
}

# --- helpers ---

check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        echo "  PASS  $label"
    else
        echo "  FAIL  $label (expected '$expected', got '$actual')"
        return 1
    fi
}

# POST /v1/license/provision — returns HTTP status code
provision_license() {
    local license_key="$1" max_seats="$2" status="$3"
    curl -sS -o /dev/null -w '%{http_code}' -X POST "$WORKER_URL/v1/license/provision" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{\"license_key\":\"$license_key\",\"max_seats\":$max_seats,\"status\":\"$status\"}"
}

# POST /v1/license/refresh — returns JSON body
refresh_license() {
    local license_key="$1" device_id="$2"
    curl -sS -X POST "$WORKER_URL/v1/license/refresh" \
        -H "Content-Type: application/json" \
        --data "{\"license_key\":\"$license_key\",\"device_id\":\"$device_id\"}"
}

# --- commands ---

cmd_provision() {
    local max_seats="$1"
    local license_key="${2:-lifecycle-$(date +%s)}"

    echo "[license-lifecycle] Provisioning license '$license_key' with $max_seats seats..."
    local http_code
    http_code="$(provision_license "$license_key" "$max_seats" "active")"
    check "provision -> 204" "204" "$http_code"
    echo "$license_key"
}

cmd_revoke() {
    local license_key="$1"

    echo "[license-lifecycle] Revoking license '$license_key'..."
    local http_code
    http_code="$(provision_license "$license_key" "1" "revoked")"
    check "revoke -> 204" "204" "$http_code"
}

cmd_status() {
    local license_key="$1"

    echo "[license-lifecycle] Checking status of license '$license_key'..."
    # We infer status by attempting a refresh — if it fails with "revoked", the license is revoked.
    # If it succeeds or fails with another reason, we report accordingly.
    local response
    response="$(refresh_license "$license_key" "status-check-device")"
    local error
    error="$(jq -r '.error // empty' <<<"$response")"
    local allowed
    allowed="$(jq -r '.allowed // empty' <<<"$response")"

    if [[ "$error" == "revoked" ]]; then
        echo "  PASS  license '$license_key' is revoked"
    elif [[ "$error" == "seat_exhausted" ]]; then
        echo "  INFO  license '$license_key' is active but seats exhausted"
    elif [[ -n "$allowed" && "$allowed" == "true" ]]; then
        echo "  INFO  license '$license_key' is active (token issued)"
    else
        echo "  INFO  license '$license_key' status: error='$error' allowed='$allowed'"
    fi
}

# --- main ---

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

COMMAND="$1"
shift

case "$COMMAND" in
    provision)
        if [[ $# -lt 1 ]]; then
            echo "provision requires <max_seats> argument" >&2
            exit 2
        fi
        cmd_provision "$@"
        ;;
    revoke)
        if [[ $# -lt 1 ]]; then
            echo "revoke requires <license_id> argument" >&2
            exit 2
        fi
        cmd_revoke "$@"
        ;;
    status)
        if [[ $# -lt 1 ]]; then
            echo "status requires <license_id> argument" >&2
            exit 2
        fi
        cmd_status "$@"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "unknown command: $COMMAND" >&2
        usage >&2
        exit 2
        ;;
esac
