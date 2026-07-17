#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ENV_FILE="$REPO_ROOT/.config/curve_copilot_full.env"

V2_CURVE_COPILOT_MODEL_FILENAME="curve_copilot_20260630_v2_k48opt.onnx"

resolve_default_model() {
    if [[ -n "${THYLLORE_CURVE_MODEL:-}" ]]; then
        printf '%s' "$THYLLORE_CURVE_MODEL"
    elif [[ -n "${THYLLORE_SHARED_DATA_DIR:-}" ]]; then
        printf '%s' "$THYLLORE_SHARED_DATA_DIR/exports/$V2_CURVE_COPILOT_MODEL_FILENAME"
    fi
}

SCENE="$REPO_ROOT/blender/test.blend"
MODEL="$(resolve_default_model)"
PLATFORM="linux_x86_64"
REBUILD_WHEEL=1
MODE="degrade"

usage() {
    cat <<EOF
Usage: $0 [--mode degrade|full|private] [options] [scene.blend]

Builds the debug Blender addon (model + ONNX Runtime + debug logging to
<repo>/log/log_blender.log), installs it, and launches Blender with the test
scene. Runtime details of each inserted keyframe are written to the log.

The wheel is rebuilt with the \`debug-log\` feature so the Rust log-output
methods are present (production wheels omit them). The actual logging
implementation lives in Rust; _debuglog.py only calls it.

Options:
  --mode MODE          Curve copilot mode, same vocabulary as ./run.sh engine
                       (degrade=A, full=B, private=C; default: degrade).
                       SSoT: crates/thyllore-ml-core/src/mode.rs
  --skip-wheel         Reuse the existing wheel (only safe if it was already
                       built with --debug-log)
  --model PATH         curve_copilot ONNX to bundle. Default resolves from
                       \$THYLLORE_CURVE_MODEL, else
                       \$THYLLORE_SHARED_DATA_DIR/exports/$V2_CURVE_COPILOT_MODEL_FILENAME.
                       Set these in your shell rc so no absolute path is hardcoded.
  -h, --help           Show this help
EOF
}

build_mode_for() {
    case "$1" in
        degrade) printf 'A' ;;
        full)    printf 'B' ;;
        private) printf 'C' ;;
        *)
            echo "invalid mode: $1 (expected degrade, full or private)" >&2
            exit 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --skip-wheel) REBUILD_WHEEL=0; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) SCENE="$1"; shift ;;
    esac
done

BUILD_MODE="$(build_mode_for "$MODE")"

if [[ -z "$MODEL" ]]; then
    echo "curve_copilot model path is not set. Pass --model PATH, or set THYLLORE_CURVE_MODEL," >&2
    echo "or set THYLLORE_SHARED_DATA_DIR (e.g. in your shell rc) so that" >&2
    echo "\$THYLLORE_SHARED_DATA_DIR/exports/$V2_CURVE_COPILOT_MODEL_FILENAME exists." >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "curve_copilot model not found at $MODEL" >&2
    exit 1
fi

export THYLLORE_BLENDER_PATH="${THYLLORE_BLENDER_PATH:-/snap/bin/blender}"
if [[ ! -x "$THYLLORE_BLENDER_PATH" ]]; then
    echo "Blender not found at $THYLLORE_BLENDER_PATH (set THYLLORE_BLENDER_PATH)" >&2
    exit 1
fi

if [[ -r "$FULL_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$FULL_ENV_FILE"
    set +a
fi
missing=()
for var in THYLLORE_FEEDBACK_TEST_ENDPOINT THYLLORE_INGEST_TOKEN; do
    [[ -z "${!var:-}" ]] && missing+=("$var")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "the addon build needs: ${missing[*]}" >&2
    echo "Fill them in once in $FULL_ENV_FILE (see './run.sh engine --help')." >&2
    exit 2
fi

if [[ "$REBUILD_WHEEL" -eq 1 ]]; then
    bash "$REPO_ROOT/scripts/collect_wheels.sh" --debug-log
fi

echo "[run_blender_debug] mode=$MODE (build mode $BUILD_MODE)"
bash "$REPO_ROOT/scripts/build_blender_addon.sh" \
    --platform "$PLATFORM" --debug --build-mode "$BUILD_MODE" \
    --env test --include-onnx-model --onnx-source-path "$MODEL"

rm -rf "$HOME"/.config/blender/*/extensions/.local/lib/python*/site-packages/thyllore_ml_core*
echo "[run_blender_debug] purged Blender-managed wheel copies (same-version wheels are not re-extracted otherwise)"

ZIP="$(cat "$REPO_ROOT/dist/.last_built_zip")"
"$THYLLORE_BLENDER_PATH" --command extension install-file -r user_default --enable "$ZIP"

echo "[run_blender_debug] launching Blender with $SCENE"
echo "[run_blender_debug] addon log -> $REPO_ROOT/log/log_blender.log"
exec "$THYLLORE_BLENDER_PATH" "$SCENE"
