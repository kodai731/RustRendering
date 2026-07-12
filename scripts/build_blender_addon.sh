#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLATFORM=""
VARIANT="lite"
BUILD_MODE="A"
VERSION="0.0.1"
OUTPUT_DIR="dist"
INCLUDE_ONNX_MODEL=0
ONNX_SOURCE_PATH="${THYLLORE_CURVE_COPILOT_ONNX:-ml/model/curve_copilot.onnx}"
ONNXRUNTIME_LIB_PATH=""
DEBUG_BUILD=0
SKIP_BLENDER_VALIDATE=0
ENDPOINT_ENV=""

usage() {
    cat <<EOF
Usage: $0 --platform PLATFORM [options]

Linux/macOS port of scripts/build_blender_addon.ps1. Builds the Blender
extension ZIP for the requested platform/variant.

Required:
  --platform PLATFORM            One of: win_amd64, linux_x86_64, macosx_arm64

Options:
  --variant lite|full            "lite" excludes Tier A (default: lite)
  --build-mode A|B|C             Distribution build mode (default: A).
                                 SSoT (behaviour, required env vars):
                                 crates/thyllore-ml-core/src/mode.rs
  --env prod|test                Feedback endpoint to bake: reads
                                 THYLLORE_FEEDBACK_PROD_ENDPOINT or
                                 THYLLORE_FEEDBACK_TEST_ENDPOINT. Without
                                 this option, THYLLORE_FEEDBACK_ENDPOINT
                                 is used as before (CI compatibility)
  --version VERSION              Extension version (default: $VERSION)
  --output-dir PATH              Output directory (default: $OUTPUT_DIR)
  --include-onnx-model           Bundle the curve_copilot ONNX
  --onnx-source-path PATH        Source path of the ONNX (default: $ONNX_SOURCE_PATH)
  --onnxruntime-lib PATH         ONNX Runtime shared library to bundle in lib/
                                 (default: vendored runtime for the platform)
  --debug                        Debug build: log addon runtime to
                                 <repo>/log/log_blender.log
  --skip-blender-validate        Skip "blender --command extension validate"
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)               PLATFORM="$2"; shift 2 ;;
        --variant)
            case "$2" in
                lite|full) VARIANT="$2" ;;
                *) echo "invalid variant: $2 (expected lite or full)" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --build-mode)
            case "$2" in
                A|B|C) BUILD_MODE="$2" ;;
                *) echo "invalid build mode: $2 (expected A, B or C)" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --env)
            case "$2" in
                prod|test) ENDPOINT_ENV="$2" ;;
                *) echo "invalid env: $2 (expected prod or test)" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --version)                VERSION="$2"; shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2"; shift 2 ;;
        --include-onnx-model)     INCLUDE_ONNX_MODEL=1; shift ;;
        --onnx-source-path)       ONNX_SOURCE_PATH="$2"; shift 2 ;;
        --onnxruntime-lib)        ONNXRUNTIME_LIB_PATH="$2"; shift 2 ;;
        --debug)                  DEBUG_BUILD=1; shift ;;
        --skip-blender-validate)  SKIP_BLENDER_VALIDATE=1; shift ;;
        -h|--help)                usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    echo "--platform is required" >&2
    usage >&2
    exit 2
fi

case "$PLATFORM" in
    win_amd64)
        BLENDER_NAME="windows-x64"
        WHEEL_MATCHERS=("win_amd64\\.whl$")
        ORT_LIB_DEST_NAME="onnxruntime.dll"
        ORT_LIB_DEFAULT=""
        ;;
    linux_x86_64)
        BLENDER_NAME="linux-x64"
        WHEEL_MATCHERS=(
            "manylinux2014_x86_64\\.whl$"
            "manylinux_2_[0-9]+_x86_64\\.whl$"
            "linux_x86_64\\.whl$"
        )
        ORT_LIB_DEST_NAME="libonnxruntime.so"
        ORT_LIB_DEFAULT="$REPO_ROOT/vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so"
        ;;
    macosx_arm64)
        BLENDER_NAME="macos-arm64"
        WHEEL_MATCHERS=(
            "macosx_[0-9]+_[0-9]+_arm64\\.whl$"
            "macosx_[0-9]+_[0-9]+_universal2\\.whl$"
        )
        ORT_LIB_DEST_NAME="libonnxruntime.dylib"
        ORT_LIB_DEFAULT=""
        ;;
    *)
        echo "invalid platform: $PLATFORM" >&2
        exit 2
        ;;
esac

case "$ENDPOINT_ENV" in
    prod) FEEDBACK_ENDPOINT_VAR="THYLLORE_FEEDBACK_PROD_ENDPOINT" ;;
    test) FEEDBACK_ENDPOINT_VAR="THYLLORE_FEEDBACK_TEST_ENDPOINT" ;;
    "")   FEEDBACK_ENDPOINT_VAR="THYLLORE_FEEDBACK_ENDPOINT" ;;
esac
FEEDBACK_ENDPOINT="${!FEEDBACK_ENDPOINT_VAR:-}"
INGEST_TOKEN="${THYLLORE_INGEST_TOKEN:-}"
FULL_TOKEN_PUBKEY="${THYLLORE_FULL_TOKEN_PUBKEY_B64:-}"
LICENSE_ENDPOINT="${THYLLORE_LICENSE_ENDPOINT:-}"

require_build_mode_env() {
    local name="$1" value="$2"
    if [[ -z "$value" ]]; then
        echo "build mode $BUILD_MODE requires environment variable $name" >&2
        exit 2
    fi
}

require_build_mode_env "$FEEDBACK_ENDPOINT_VAR" "$FEEDBACK_ENDPOINT"
require_build_mode_env THYLLORE_INGEST_TOKEN "$INGEST_TOKEN"
echo "[build_blender_addon] Feedback endpoint (${ENDPOINT_ENV:-explicit}): $FEEDBACK_ENDPOINT"

case "$BUILD_MODE" in
    B)
        require_build_mode_env THYLLORE_FULL_TOKEN_PUBKEY_B64 "$FULL_TOKEN_PUBKEY"
        ;;
    C)
        require_build_mode_env THYLLORE_LICENSE_ENDPOINT "$LICENSE_ENDPOINT"
        require_build_mode_env THYLLORE_FULL_TOKEN_PUBKEY_B64 "$FULL_TOKEN_PUBKEY"
        ;;
esac

echo "[build_blender_addon] Platform: $PLATFORM -> Blender: $BLENDER_NAME, variant: $VARIANT, build mode: $BUILD_MODE"

STAGE_DIR="$REPO_ROOT/build/blender_addon_stage_${PLATFORM}_${VARIANT}_${BUILD_MODE}"
SOURCE_DIR="$REPO_ROOT/blender_addon"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

if command -v rsync >/dev/null 2>&1; then
    rsync -a \
        --exclude='tests/' \
        --exclude='__pycache__/' \
        --exclude='.pytest_cache/' \
        --exclude='.egg-info/' \
        --exclude='wheels-extracted/' \
        --exclude='*.pyc' \
        "$SOURCE_DIR/" "$STAGE_DIR/"
else
    cp -a "$SOURCE_DIR/." "$STAGE_DIR/"
    find "$STAGE_DIR" -type d \( -name tests -o -name __pycache__ -o -name .pytest_cache -o -name .egg-info -o -name wheels-extracted \) -prune -exec rm -rf {} +
    find "$STAGE_DIR" -type f -name "*.pyc" -delete
fi

LITE_EXCLUDE_REL_PATHS=(
    "operators/auto_rig.py"
    "operators/text_to_mesh.py"
    "operators/_grpc_helpers.py"
    "grpc_client"
    "blender_manifest.toml"
    "operators/text_to_motion.py"
)

if [[ "$VARIANT" == "lite" ]]; then
    for rel in "${LITE_EXCLUDE_REL_PATHS[@]}"; do
        abs="$STAGE_DIR/$rel"
        if [[ -e "$abs" ]]; then
            rm -rf "$abs"
            echo "[build_blender_addon] (lite) excluded: $rel"
        fi
    done
else
    rm -f "$STAGE_DIR/blender_manifest.lite.toml"
fi

if [[ "$BUILD_MODE" != "B" && -e "$STAGE_DIR/telemetry" ]]; then
    rm -rf "$STAGE_DIR/telemetry"
    echo "[build_blender_addon] (mode $BUILD_MODE) excluded: telemetry"
fi
if [[ "$BUILD_MODE" != "C" && -e "$STAGE_DIR/license_client" ]]; then
    rm -rf "$STAGE_DIR/license_client"
    echo "[build_blender_addon] (mode $BUILD_MODE) excluded: license_client"
fi

WHEELS_DIR="$STAGE_DIR/wheels"
if [[ ! -d "$WHEELS_DIR" ]]; then
    echo "Stage directory has no wheels/. Run scripts/collect_wheels.sh first." >&2
    exit 1
fi

LITE_ALLOWED_WHEEL_PREFIXES=("thyllore_ml_core")

wheel_matches_platform() {
    local name="$1"
    if [[ "$name" =~ py3-none-any\.whl$ ]]; then
        return 0
    fi
    for pattern in "${WHEEL_MATCHERS[@]}"; do
        if [[ "$name" =~ $pattern ]]; then
            return 0
        fi
    done
    return 1
}

wheel_allowed_for_lite() {
    local name="$1"
    for prefix in "${LITE_ALLOWED_WHEEL_PREFIXES[@]}"; do
        if [[ "$name" == "${prefix}"* ]]; then
            return 0
        fi
    done
    return 1
}

KEPT_WHEELS=()
shopt -s nullglob
for wheel_path in "$WHEELS_DIR"/*.whl; do
    name="$(basename "$wheel_path")"

    if [[ "$VARIANT" == "lite" ]] && ! wheel_allowed_for_lite "$name"; then
        rm -f "$wheel_path"
        continue
    fi

    if wheel_matches_platform "$name"; then
        KEPT_WHEELS+=("$name")
    else
        rm -f "$wheel_path"
    fi
done
shopt -u nullglob

rm -f "$WHEELS_DIR/HASHES.txt" "$WHEELS_DIR/README.md"

MIN_WHEELS=4
[[ "$VARIANT" == "lite" ]] && MIN_WHEELS=1
if [[ ${#KEPT_WHEELS[@]} -lt $MIN_WHEELS ]]; then
    echo "Expected at least $MIN_WHEELS wheels for $PLATFORM/$VARIANT, got ${#KEPT_WHEELS[@]} (found: ${KEPT_WHEELS[*]:-none})" >&2
    exit 1
fi
echo "[build_blender_addon] Kept ${#KEPT_WHEELS[@]} wheels for $PLATFORM/$VARIANT"

CANONICAL_MANIFEST="$STAGE_DIR/blender_manifest.toml"
LITE_MANIFEST="$STAGE_DIR/blender_manifest.lite.toml"

if [[ "$VARIANT" == "lite" ]]; then
    if [[ ! -f "$LITE_MANIFEST" ]]; then
        echo "blender_manifest.lite.toml missing from stage; expected at $LITE_MANIFEST" >&2
        exit 1
    fi
    rm -f "$CANONICAL_MANIFEST"
    mv "$LITE_MANIFEST" "$CANONICAL_MANIFEST"
fi

MANIFEST_PATH="$CANONICAL_MANIFEST"

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python3 not found on PATH (set PYTHON=... to override)" >&2
    exit 1
fi

WHEEL_LINES_JOINED=""
for w in "${KEPT_WHEELS[@]}"; do
    WHEEL_LINES_JOINED+="    \"./wheels/$w\",\n"
done

WHEEL_LINES_JOINED="$WHEEL_LINES_JOINED" \
MANIFEST_PATH="$MANIFEST_PATH" \
BLENDER_NAME="$BLENDER_NAME" \
"$PYTHON_BIN" - <<'PY'
import os, re
manifest_path = os.environ["MANIFEST_PATH"]
blender_name = os.environ["BLENDER_NAME"]
wheel_lines = os.environ["WHEEL_LINES_JOINED"].replace("\\n", "\n").rstrip("\n")

with open(manifest_path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8-sig")

text = text.replace("PLATFORM_BLENDER_NAME", blender_name)
text = re.sub(
    r"wheels = \[.*?\]",
    f"wheels = [\n{wheel_lines}\n]",
    text,
    flags=re.DOTALL,
)

with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    f.write(text)
PY

EXPECTED_ABI="$("$PYTHON_BIN" -c "
import re, pathlib
m = re.search(
    r'EXPECTED_ABI_MARKER\s*[:=]\s*int\s*=\s*(\d+)',
    pathlib.Path('$REPO_ROOT/blender_addon/__init__.py').read_text(encoding='utf-8'),
)
print(m.group(1) if m else 'MISSING')
")"
if [[ "$EXPECTED_ABI" == "MISSING" ]]; then
    echo "EXPECTED_ABI_MARKER not found in blender_addon/__init__.py" >&2
    exit 1
fi

API_MARKER="$("$PYTHON_BIN" -c "
import re, pathlib
m = re.search(
    r'pub const ABI_MARKER:\s*u32\s*=\s*(\d+)',
    pathlib.Path('$REPO_ROOT/crates/thyllore-ml-api/src/lib.rs').read_text(encoding='utf-8'),
)
print(m.group(1) if m else 'MISSING')
")"
if [[ "$API_MARKER" == "MISSING" ]]; then
    echo "ABI_MARKER not found in crates/thyllore-ml-api/src/lib.rs" >&2
    exit 1
fi

if [[ "${EXPECTED_ABI// /}" != "${API_MARKER// /}" ]]; then
    echo "ABI marker mismatch: addon EXPECTED_ABI_MARKER=$EXPECTED_ABI vs ml-api ABI_MARKER=$API_MARKER. Bump both or neither." >&2
    exit 1
fi
echo "[build_blender_addon] ABI marker verified: $API_MARKER"

if [[ "$INCLUDE_ONNX_MODEL" -eq 1 ]]; then
    if [[ "$ONNX_SOURCE_PATH" = /* ]]; then
        ONNX_ABS="$ONNX_SOURCE_PATH"
    else
        ONNX_ABS="$REPO_ROOT/$ONNX_SOURCE_PATH"
    fi
    if [[ ! -f "$ONNX_ABS" ]]; then
        echo "ONNX model not found at $ONNX_ABS (omit --include-onnx-model to skip)" >&2
        exit 1
    fi
    mkdir -p "$STAGE_DIR/models"
    cp -f "$ONNX_ABS" "$STAGE_DIR/models/curve_copilot.onnx"
    echo "[build_blender_addon] Bundled ONNX model from $ONNX_SOURCE_PATH"
fi

if [[ -z "$ONNXRUNTIME_LIB_PATH" ]]; then
    ONNXRUNTIME_LIB_PATH="$ORT_LIB_DEFAULT"
fi
if [[ -n "$ONNXRUNTIME_LIB_PATH" && -f "$ONNXRUNTIME_LIB_PATH" ]]; then
    mkdir -p "$STAGE_DIR/lib"
    cp -L -f "$ONNXRUNTIME_LIB_PATH" "$STAGE_DIR/lib/$ORT_LIB_DEST_NAME"
    echo "[build_blender_addon] Bundled ONNX Runtime: $ONNXRUNTIME_LIB_PATH -> lib/$ORT_LIB_DEST_NAME"
else
    echo "[build_blender_addon] WARNING: ONNX Runtime lib not bundled for $PLATFORM" >&2
    echo "  (in-process inference will fail unless the user sets ORT_DYLIB_PATH)" >&2
    echo "  searched: ${ONNXRUNTIME_LIB_PATH:-<none>}" >&2
fi

BUILD_CONFIG_PATH="$STAGE_DIR/build_config.py"
{
    echo "BUILD_MODE = \"$BUILD_MODE\""
    echo "FEEDBACK_ENDPOINT = \"$FEEDBACK_ENDPOINT\""
    echo "INGEST_TOKEN = \"$INGEST_TOKEN\""
    if [[ "$BUILD_MODE" == "B" || "$BUILD_MODE" == "C" ]]; then
        echo "FULL_TOKEN_PUBKEY = \"$FULL_TOKEN_PUBKEY\""
    fi
    if [[ "$BUILD_MODE" == "C" ]]; then
        echo "LICENSE_ENDPOINT = \"$LICENSE_ENDPOINT\""
    fi
} >"$BUILD_CONFIG_PATH"
echo "[build_blender_addon] Generated build_config.py (mode $BUILD_MODE)"

DEBUG_CONFIG_PATH="$STAGE_DIR/_debug_config.py"
if [[ "$DEBUG_BUILD" -eq 1 ]]; then
    DEBUG_LOG_PATH="$REPO_ROOT/log/log_blender.log"
    {
        echo "DEBUG_BUILD = True"
        echo "LOG_PATH = r\"$DEBUG_LOG_PATH\""
    } >"$DEBUG_CONFIG_PATH"
    echo "[build_blender_addon] Debug build: logging to $DEBUG_LOG_PATH"
else
    rm -f "$DEBUG_CONFIG_PATH"
fi

resolve_blender_executable() {
    if [[ -n "${THYLLORE_BLENDER_PATH:-}" && -x "$THYLLORE_BLENDER_PATH" ]]; then
        echo "$THYLLORE_BLENDER_PATH"
        return
    fi
    for candidate in /usr/bin/blender /opt/blender/blender /snap/bin/blender; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

if [[ "$SKIP_BLENDER_VALIDATE" -ne 1 ]]; then
    BLENDER_EXE="$(resolve_blender_executable)"
    if [[ -n "$BLENDER_EXE" ]]; then
        echo "[build_blender_addon] Validating with $BLENDER_EXE..."
        "$BLENDER_EXE" --command extension validate "$STAGE_DIR"
    else
        echo "[build_blender_addon] Blender executable not found - skipping validate" >&2
    fi
fi

ABS_OUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
mkdir -p "$ABS_OUT_DIR"

if [[ "$VARIANT" == "lite" ]]; then
    ZIP_BASENAME="thyllore_animation_curve_copilot"
else
    ZIP_BASENAME="thyllore_animation_addon"
fi
case "$BUILD_MODE" in
    A) ZIP_BASENAME="${ZIP_BASENAME}_degraded" ;;
    B) ZIP_BASENAME="${ZIP_BASENAME}_full" ;;
    C) ZIP_BASENAME="${ZIP_BASENAME}_private" ;;
esac
if [[ "$ENDPOINT_ENV" == "test" ]]; then
    ZIP_BASENAME="${ZIP_BASENAME}_test"
fi
ZIP_PATH="$ABS_OUT_DIR/${ZIP_BASENAME}-${VERSION}-${PLATFORM}.zip"
rm -f "$ZIP_PATH"

(
    cd "$STAGE_DIR"
    zip -q -r -X "$ZIP_PATH" .
)

HASH="$(sha256sum "$ZIP_PATH" | awk '{print $1}')"
SIZE_BYTES="$(stat -c '%s' "$ZIP_PATH" 2>/dev/null || stat -f '%z' "$ZIP_PATH")"
SIZE_MB="$(awk -v b="$SIZE_BYTES" 'BEGIN{printf "%.2f", b/1048576}')"

echo
echo "[build_blender_addon] Created: $ZIP_PATH"
echo "[build_blender_addon] Size:    $SIZE_MB MiB"
echo "[build_blender_addon] SHA256:  $HASH"

SENTINEL_PATH="$ABS_OUT_DIR/.last_built_zip"
printf '%s' "$ZIP_PATH" >"$SENTINEL_PATH"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
        echo "zip_path=$ZIP_PATH"
        echo "zip_name=$(basename "$ZIP_PATH")"
    } >>"$GITHUB_OUTPUT"
fi

rm -rf "$STAGE_DIR"
