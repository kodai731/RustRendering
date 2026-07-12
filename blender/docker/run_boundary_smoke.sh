#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="thyllore-blender-smoke"
BLENDER_VERSION="${THYLLORE_SMOKE_BLENDER_VERSION:-5.1.2}"
MODE="degrade"
ZIP=""
ONNX=""
REINSTALL_BLENDER=0

usage() {
    cat <<EOF
Usage: $0 [--mode degrade|full|private] [--zip PATH] [--blender-version X.Y.Z]
          [--reinstall-blender] [-- <extra build_mode_boundary_smoke.py args>]

Runs the layer-3 boundary smoke against a production ZIP inside a Docker
container with a pristine Blender (no pre-installed addons, fresh user
profile on every run).

  --mode MODE            degrade=A, full=B, private=C (default: degrade)
  --zip PATH             extension ZIP to test. Default resolves from dist/
                         by mode (thyllore_animation_curve_copilot_<mode>-*.zip)
  --blender-version VER  Blender to install in the image (default: $BLENDER_VERSION)
  --reinstall-blender    rebuild the image without cache (re-downloads Blender)
  --onnx PATH            curve_copilot ONNX to mount for the operator run
                         (required by -- --live-send)
  -- ARGS                passed through to build_mode_boundary_smoke.py
                         (e.g. -- --live-send)
EOF
}

expect_mode_for() {
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

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --zip) ZIP="$2"; shift 2 ;;
        --onnx) ONNX="$2"; shift 2 ;;
        --blender-version) BLENDER_VERSION="$2"; shift 2 ;;
        --reinstall-blender) REINSTALL_BLENDER=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

EXPECT_MODE="$(expect_mode_for "$MODE")"
if [[ -z "$ZIP" ]]; then
    ZIP="$REPO_ROOT/dist/thyllore_animation_curve_copilot_${MODE/degrade/degraded}-0.0.1-linux_x86_64.zip"
fi
if [[ ! -f "$ZIP" ]]; then
    echo "extension ZIP not found: $ZIP (build it with scripts/build_blender_addon.sh)" >&2
    exit 1
fi

BUILD_FLAGS=(--build-arg "BLENDER_VERSION=$BLENDER_VERSION")
if [[ "$REINSTALL_BLENDER" -eq 1 ]]; then
    BUILD_FLAGS+=(--no-cache)
fi
docker build "${BUILD_FLAGS[@]}" -t "$IMAGE_TAG" "$REPO_ROOT/blender/docker"

OUT_DIR="$REPO_ROOT/build/docker_smoke"
mkdir -p "$OUT_DIR"
ZIP_DIR="$(cd "$(dirname "$ZIP")" && pwd)"
ZIP_NAME="$(basename "$ZIP")"

ONNX_MOUNT=()
if [[ -n "$ONNX" ]]; then
    if ! ONNX_ABS="$(realpath -e "$ONNX" 2>/dev/null)"; then
        echo "onnx model not found: $ONNX" >&2
        exit 1
    fi
    ONNX_MOUNT=(-v "$ONNX_ABS:/onnx/curve_copilot.onnx:ro")
    EXTRA_ARGS+=(--onnx /onnx/curve_copilot.onnx)
fi

docker run --rm \
    -v "$REPO_ROOT:/workspace:ro" \
    -v "$ZIP_DIR:/zips:ro" \
    -v "$OUT_DIR:/out" \
    ${ONNX_MOUNT[@]+"${ONNX_MOUNT[@]}"} \
    "$IMAGE_TAG" \
    bash /workspace/blender/docker/smoke_entrypoint.sh \
    "/zips/$ZIP_NAME" "$EXPECT_MODE" "/out/result_${EXPECT_MODE}.json" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[run_boundary_smoke] result -> $OUT_DIR/result_${EXPECT_MODE}.json"
