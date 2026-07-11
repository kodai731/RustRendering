#!/usr/bin/env bash
set -euo pipefail

# Deploys the Curve Copilot feedback Worker to Cloudflare using only curl + jq
# (no npm, no wrangler). Reads non-secret config from wrangler.toml as the
# single source of truth; secrets come from the environment and are never
# printed or written to disk unencrypted.
#
# Required environment:
#   CF_API_TOKEN            Cloudflare API token (scopes: Workers Scripts:Edit,
#                           Workers R2 Storage:Edit, Account Settings:Read)
#   CF_ACCOUNT_ID           Cloudflare account id
#   THYLLORE_INGEST_TOKEN   shared bearer token baked into mode B addon builds
#   THYLLORE_ADMIN_TOKEN    bearer token guarding /v1/license/provision
#   Ed25519 private key (PKCS8 DER, base64), via EITHER:
#     THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE   path to the b64 file
#     THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64         the b64 value itself
#   (scripts/gen_license_keypair.sh writes secrets/private_key_pkcs8.b64)

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRANGLER_TOML="$WORKER_DIR/wrangler.toml"
API_BASE="https://api.cloudflare.com/client/v4"

SKIP_BUCKET=0
DRY_RUN=0
ENVIRONMENT="prod"

usage() {
    cat <<EOF
Usage: $0 [options]

Deploys worker/src/index.mjs to Cloudflare via the REST API (curl only).

Options:
  --env prod|test  Target environment (default: prod). "test" appends a "-test"
                   suffix to both the worker name and the R2 bucket, keeping the
                   test data fully separate from production.
  --skip-bucket    Do not attempt to create the R2 bucket
  --dry-run        Build and validate everything but do not call the upload API
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            case "$2" in
                prod|test) ENVIRONMENT="$2" ;;
                *) echo "invalid env: $2 (expected prod or test)" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --skip-bucket) SKIP_BUCKET=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for tool in curl jq; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "required tool not found: $tool" >&2
        exit 1
    fi
done

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "environment variable $name is required" >&2
        exit 2
    fi
}

require_env CF_API_TOKEN
require_env CF_ACCOUNT_ID
require_env THYLLORE_INGEST_TOKEN
require_env THYLLORE_ADMIN_TOKEN

resolve_private_key() {
    if [[ -n "${THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE:-}" ]]; then
        if [[ ! -f "$THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE" ]]; then
            echo "private key file not found: $THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE" >&2
            exit 2
        fi
        tr -d '\r\n' <"$THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE"
        return
    fi
    if [[ -n "${THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64:-}" ]]; then
        printf '%s' "$THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64" | tr -d '\r\n'
        return
    fi
    echo "provide THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64_FILE or THYLLORE_UNLOCK_PRIVATE_KEY_PKCS8_B64" >&2
    exit 2
}

toml_scalar() {
    local key="$1"
    sed -n -E "s/^${key}[[:space:]]*=[[:space:]]*\"([^\"]*)\".*/\1/p" "$WRANGLER_TOML" | head -n1
}

toml_r2_field() {
    local field="$1"
    awk -F'"' -v field="$field" '
        /^\[\[r2_buckets\]\]/ { in_block = 1; next }
        /^\[/ { in_block = 0 }
        in_block && $0 ~ ("^" field "[[:space:]]*=") { print $2; exit }
    ' "$WRANGLER_TOML"
}

toml_section_field() {
    local section="$1" field="$2"
    awk -F'"' -v section="$section" -v field="$field" '
        index($0, "[[" section "]]") == 1 { in_block = 1; next }
        /^\[/ { in_block = 0 }
        in_block && $0 ~ ("^" field "[[:space:]]*=") { print $2; exit }
    ' "$WRANGLER_TOML"
}

build_var_bindings() {
    local vars_array="[]"
    local in_vars=0 line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        case "$line" in
            '[vars]') in_vars=1; continue ;;
            '['*)     in_vars=0 ;;
        esac
        if [[ $in_vars -eq 1 && "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*\"(.*)\"[[:space:]]*$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            vars_array="$(jq -c --arg n "$key" --arg t "$value" \
                '. + [{type: "plain_text", name: $n, text: $t}]' <<<"$vars_array")"
        fi
    done <"$WRANGLER_TOML"
    printf '%s' "$vars_array"
}

SCRIPT_NAME="$(toml_scalar name)"
COMPAT_DATE="$(toml_scalar compatibility_date)"
R2_BINDING="$(toml_r2_field binding)"
BUCKET_NAME="$(toml_r2_field bucket_name)"
DO_BINDING="$(toml_section_field "durable_objects.bindings" name)"
DO_CLASS="$(toml_section_field "durable_objects.bindings" class_name)"
MIGRATION_TAG="$(toml_section_field migrations tag)"
MIGRATION_SQLITE_CLASS="$(toml_section_field migrations new_sqlite_classes)"
MODULE_NAME="index.mjs"
SCRIPT_PATH="$WORKER_DIR/src/$MODULE_NAME"

for value in SCRIPT_NAME COMPAT_DATE R2_BINDING BUCKET_NAME DO_BINDING DO_CLASS \
    MIGRATION_TAG MIGRATION_SQLITE_CLASS; do
    if [[ -z "${!value}" ]]; then
        echo "failed to read $value from $WRANGLER_TOML" >&2
        exit 1
    fi
done
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "worker module not found: $SCRIPT_PATH" >&2
    exit 1
fi

if [[ "$ENVIRONMENT" == "test" ]]; then
    SCRIPT_NAME="${SCRIPT_NAME}-test"
    BUCKET_NAME="${BUCKET_NAME}-test"
fi

cf_api() {
    local method="$1" path="$2" data="${3:-}"
    local args=(-sS -X "$method" "$API_BASE$path"
        -H "Authorization: Bearer $CF_API_TOKEN")
    if [[ -n "$data" ]]; then
        args+=(-H "Content-Type: application/json" --data "$data")
    fi
    curl "${args[@]}"
}

fail_on_api_error() {
    local response="$1" action="$2"
    if [[ "$(jq -r '.success' <<<"$response")" != "true" ]]; then
        echo "Cloudflare API error during $action:" >&2
        jq -r '.errors // [] | .[] | "  [\(.code)] \(.message)"' <<<"$response" >&2
        exit 1
    fi
}

echo "[deploy] Env: $ENVIRONMENT  worker: $SCRIPT_NAME  bucket: $BUCKET_NAME  compat: $COMPAT_DATE"

PRIVATE_KEY_B64="$(resolve_private_key)"

VAR_BINDINGS="$(build_var_bindings)"
METADATA="$(jq -nc \
    --arg main "$MODULE_NAME" \
    --arg compat "$COMPAT_DATE" \
    --arg r2name "$R2_BINDING" \
    --arg bucket "$BUCKET_NAME" \
    --arg doname "$DO_BINDING" \
    --arg doclass "$DO_CLASS" \
    --arg ingest "$THYLLORE_INGEST_TOKEN" \
    --arg admin "$THYLLORE_ADMIN_TOKEN" \
    --arg privkey "$PRIVATE_KEY_B64" \
    --argjson vars "$VAR_BINDINGS" \
    '{
        main_module: $main,
        compatibility_date: $compat,
        bindings: (
            [
                {type: "r2_bucket", name: $r2name, bucket_name: $bucket},
                {type: "durable_object_namespace", name: $doname, class_name: $doclass}
            ]
            + $vars
            + [
                {type: "secret_text", name: "INGEST_TOKEN", text: $ingest},
                {type: "secret_text", name: "ADMIN_TOKEN", text: $admin},
                {type: "secret_text", name: "UNLOCK_PRIVATE_KEY_PKCS8_B64", text: $privkey}
            ]
        )
    }')"

echo "[deploy] Bindings (secret values hidden):"
jq -r '.bindings[] | if .type | startswith("secret") then "  \(.type)  \(.name)  <hidden>" else "  \(.type)  \(.name)  \(.bucket_name // .class_name // .text)" end' <<<"$METADATA"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[deploy] --dry-run: config valid, no Cloudflare API calls made."
    exit 0
fi

echo "[deploy] Verifying API token..."
token_check="$(cf_api GET "/accounts/$CF_ACCOUNT_ID/tokens/verify")"
fail_on_api_error "$token_check" "token verification"

if [[ $SKIP_BUCKET -eq 0 ]]; then
    echo "[deploy] Ensuring R2 bucket '$BUCKET_NAME' exists..."
    bucket_response="$(cf_api POST "/accounts/$CF_ACCOUNT_ID/r2/buckets" \
        "$(jq -nc --arg name "$BUCKET_NAME" '{name: $name}')")"
    if [[ "$(jq -r '.success' <<<"$bucket_response")" != "true" ]]; then
        if jq -e '.errors // [] | any(.code == 10004)' <<<"$bucket_response" >/dev/null; then
            echo "[deploy] Bucket already exists, continuing."
        else
            fail_on_api_error "$bucket_response" "R2 bucket creation"
        fi
    else
        echo "[deploy] Bucket created."
    fi
fi

upload_worker() {
    local metadata="$1" meta_file response
    meta_file="$(mktemp)"
    chmod 600 "$meta_file"
    printf '%s' "$metadata" >"$meta_file"
    response="$(curl -sS -X PUT \
        "$API_BASE/accounts/$CF_ACCOUNT_ID/workers/scripts/$SCRIPT_NAME" \
        -H "Authorization: Bearer $CF_API_TOKEN" \
        -F "metadata=@$meta_file;type=application/json" \
        -F "$MODULE_NAME=@$SCRIPT_PATH;filename=$MODULE_NAME;type=application/javascript+module")"
    rm -f "$meta_file"
    printf '%s' "$response"
}

echo "[deploy] Uploading worker script..."
upload_response="$(upload_worker "$METADATA")"
if [[ "$(jq -r '.success' <<<"$upload_response")" != "true" ]]; then
    echo "[deploy] Plain upload failed (expected on first deploy of a new DO class):" >&2
    jq -r '.errors // [] | .[] | "  [\(.code)] \(.message)"' <<<"$upload_response" >&2
    echo "[deploy] Retrying with DO migration '$MIGRATION_TAG' ($MIGRATION_SQLITE_CLASS)..."
    metadata_with_migrations="$(jq -c \
        --arg tag "$MIGRATION_TAG" \
        --arg class "$MIGRATION_SQLITE_CLASS" \
        '. + {migrations: {new_tag: $tag, new_sqlite_classes: [$class]}}' <<<"$METADATA")"
    upload_response="$(upload_worker "$metadata_with_migrations")"
    fail_on_api_error "$upload_response" "worker upload (with migrations)"
fi
echo "[deploy] Worker uploaded."

echo "[deploy] Enabling workers.dev route..."
subdomain_enable="$(cf_api POST \
    "/accounts/$CF_ACCOUNT_ID/workers/scripts/$SCRIPT_NAME/subdomain" \
    "$(jq -nc '{enabled: true}')")"
fail_on_api_error "$subdomain_enable" "workers.dev route enable"

account_subdomain="$(cf_api GET "/accounts/$CF_ACCOUNT_ID/workers/subdomain")"
fail_on_api_error "$account_subdomain" "account subdomain lookup"
SUBDOMAIN="$(jq -r '.result.subdomain' <<<"$account_subdomain")"

echo
echo "[deploy] Deployed: https://$SCRIPT_NAME.$SUBDOMAIN.workers.dev"
echo "[deploy] Endpoints: POST /v1/feedback  /v1/message  /v1/license/refresh  /v1/license/provision"
echo "[deploy] Bake THYLLORE_FEEDBACK_ENDPOINT into mode B builds:"
echo "           https://$SCRIPT_NAME.$SUBDOMAIN.workers.dev/v1/feedback"
