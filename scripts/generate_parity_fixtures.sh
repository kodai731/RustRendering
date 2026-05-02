#!/usr/bin/env bash
set -euo pipefail

if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "cygwin" || -n "${MSYSTEM:-}" ]]; then
    echo "ERROR: This script must be run from WSL2 bash, not Git Bash / MSYS2 / Cygwin." >&2
    echo "       MSYS path translation creates broken directories like 'C:\\Program Files\\Git\\home\\...'" >&2
    echo "       under the workspace root. Re-run from WSL2:" >&2
    echo "         wsl bash scripts/generate_parity_fixtures.sh \"\$@\"" >&2
    exit 1
fi

ONNX_REPO="kodai731/thyllore-curve-copilot"
ONNX_REVISION="${THYLLORE_ONNX_REVISION:-main}"
ONNX_FILENAME="curve_copilot.onnx"

FORCE_DOWNLOAD=0
FIXTURE_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE_DOWNLOAD=1; shift ;;
        --fixture-root) FIXTURE_ROOT="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--force] [--fixture-root PATH]

Regenerates the ml_parity fixture set into <fixture-root> by:
  1. Downloading the curve_copilot ONNX model from HuggingFace
     ($ONNX_REPO @ $ONNX_REVISION) unless already cached
  2. Running the Rust fixture generators (cargo test --ignored) which produce
     proto/*.bin and numpy/*.json from synthetic seeds + ONNX inference
  3. Writing manifest.json with sha256 + size for every regenerated file

Default fixture root: <workspace>/fixtures/ml_parity (override with
THYLLORE_PARITY_FIXTURE_OUTPUT or --fixture-root).
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$FIXTURE_ROOT" ]]; then
    FIXTURE_ROOT="${THYLLORE_PARITY_FIXTURE_OUTPUT:-$WORKSPACE_ROOT/fixtures/ml_parity}"
fi
echo "fixture root: $FIXTURE_ROOT"
mkdir -p "$FIXTURE_ROOT/proto" "$FIXTURE_ROOT/onnx" "$FIXTURE_ROOT/numpy"

ONNX_LOCAL="$FIXTURE_ROOT/onnx/$ONNX_FILENAME"
ONNX_URL="https://huggingface.co/$ONNX_REPO/resolve/$ONNX_REVISION/$ONNX_FILENAME"

if [[ "$FORCE_DOWNLOAD" -eq 1 || ! -f "$ONNX_LOCAL" ]]; then
    echo "downloading ONNX: $ONNX_URL -> $ONNX_LOCAL"
    curl -fL --retry 3 --retry-delay 2 -o "$ONNX_LOCAL.tmp" "$ONNX_URL"
    mv -f "$ONNX_LOCAL.tmp" "$ONNX_LOCAL"
else
    echo "ONNX already present at $ONNX_LOCAL (use --force to re-download)"
fi

export THYLLORE_PARITY_FIXTURE_OUTPUT="$FIXTURE_ROOT"

echo "==> generating gRPC proto fixtures"
(
    cd "$WORKSPACE_ROOT"
    CARGO_TARGET_DIR="$WORKSPACE_ROOT/target-linux" \
        cargo test -p thyllore-grpc-client --features auto-rig,text-to-motion \
        --test grpc_fixture_generator generate_grpc_request_response_fixtures \
        -- --ignored --nocapture
)

echo "==> generating curve_copilot input + golden fixtures"
# Run native cargo; ort needs libonnxruntime.so / .dll to load via ORT_DYLIB_PATH.
# vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so must be
# extracted before running on Linux/WSL2 (one-time setup, see docs).
ORT_DYLIB_VENDOR_LINUX="$WORKSPACE_ROOT/vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so"
if [[ -z "${ORT_DYLIB_PATH:-}" && -f "$ORT_DYLIB_VENDOR_LINUX" ]]; then
    export ORT_DYLIB_PATH="$ORT_DYLIB_VENDOR_LINUX"
fi
(
    cd "$WORKSPACE_ROOT"
    CARGO_TARGET_DIR="$WORKSPACE_ROOT/target-linux" \
        cargo test -p thyllore-ml-core --test curve_copilot_fixture_generator \
        generate_curve_copilot_input_and_golden_fixtures -- --ignored --nocapture
)

unset THYLLORE_PARITY_FIXTURE_OUTPUT

COMMIT=$(cd "$WORKSPACE_ROOT" && git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
GENERATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "==> writing manifest.json"
python3 - "$FIXTURE_ROOT" "$COMMIT" "$GENERATED_AT" "$ONNX_REVISION" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

fixture_root = Path(sys.argv[1])
commit = sys.argv[2]
generated_at = sys.argv[3]
onnx_revision = sys.argv[4]

manifest = {
    "schema_version": 1,
    "generated_at": generated_at,
    "generator": "scripts/generate_parity_fixtures.sh",
    "thyllore_animation_commit": commit,
    "onnx_huggingface_revision": onnx_revision,
    "proto_version": "v1",
    "fixtures": {},
}

excluded_filenames = {"manifest.json", "README.md", ".gitkeep"}
for path in sorted(fixture_root.rglob("*")):
    if not path.is_file() or path.name in excluded_filenames:
        continue
    rel = str(path.relative_to(fixture_root)).replace(os.sep, "/")
    data = path.read_bytes()
    manifest["fixtures"][rel] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

(fixture_root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(f"manifest written: {fixture_root}/manifest.json ({len(manifest['fixtures'])} entries)")
PY

echo
echo "fixtures regenerated at $FIXTURE_ROOT"
echo "ONNX revision: $ONNX_REVISION"
