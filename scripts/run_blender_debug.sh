#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAWFUTURE_MODEL_FILENAME="curve_copilot_20260531_rawfuture_v1_tangent.onnx"

resolve_default_model() {
    if [[ -n "${THYLLORE_CURVE_MODEL:-}" ]]; then
        printf '%s' "$THYLLORE_CURVE_MODEL"
    elif [[ -n "${THYLLORE_SHARED_DATA_DIR:-}" ]]; then
        printf '%s' "$THYLLORE_SHARED_DATA_DIR/exports/$RAWFUTURE_MODEL_FILENAME"
    fi
}

SCENE="$REPO_ROOT/blender/test.blend"
MODEL="$(resolve_default_model)"
PLATFORM="linux_x86_64"
REBUILD_WHEEL=1

usage() {
    cat <<EOF
Usage: $0 [options] [scene.blend]

Builds the debug Blender addon (model + ONNX Runtime + debug logging to
<repo>/log/log_blender.log), installs it, and launches Blender with the test
scene. Runtime details of each inserted keyframe are written to the log.

The wheel is rebuilt with the \`debug-log\` feature so the Rust log-output
methods are present (production wheels omit them). The actual logging
implementation lives in Rust; _debuglog.py only calls it.

Options:
  --skip-wheel         Reuse the existing wheel (only safe if it was already
                       built with --debug-log)
  --model PATH         curve_copilot ONNX to bundle. Default resolves from
                       \$THYLLORE_CURVE_MODEL, else
                       \$THYLLORE_SHARED_DATA_DIR/exports/$RAWFUTURE_MODEL_FILENAME.
                       Set these in your shell rc so no absolute path is hardcoded.
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-wheel) REBUILD_WHEEL=0; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) SCENE="$1"; shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "curve_copilot model path is not set. Pass --model PATH, or set THYLLORE_CURVE_MODEL," >&2
    echo "or set THYLLORE_SHARED_DATA_DIR (e.g. in your shell rc) so that" >&2
    echo "\$THYLLORE_SHARED_DATA_DIR/exports/$RAWFUTURE_MODEL_FILENAME exists." >&2
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

if [[ "$REBUILD_WHEEL" -eq 1 ]]; then
    bash "$REPO_ROOT/scripts/collect_wheels.sh" --skip-pip-download --variant lite --debug-log
fi

bash "$REPO_ROOT/scripts/build_blender_addon.sh" \
    --platform "$PLATFORM" --variant lite --debug \
    --include-onnx-model --onnx-source-path "$MODEL"

ZIP="$REPO_ROOT/dist/thyllore_animation_lite-0.0.1-${PLATFORM}.zip"
"$THYLLORE_BLENDER_PATH" --command extension install-file -r user_default --enable "$ZIP"

echo "[run_blender_debug] launching Blender with $SCENE"
echo "[run_blender_debug] addon log -> $REPO_ROOT/log/log_blender.log"
exec "$THYLLORE_BLENDER_PATH" "$SCENE"
