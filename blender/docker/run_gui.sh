#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="thyllore-blender-smoke"
BLENDER_VERSION="${THYLLORE_SMOKE_BLENDER_VERSION:-5.1.2}"
MODE="degrade"
ZIP=""
SCENE=""
SOFTWARE_GL=0
NO_INSTALL=0
PYTHON_SCRIPT=""

usage() {
    cat <<EOF
Usage: $0 [--mode degrade|full|private] [--zip PATH] [--scene PATH.blend]
          [--blender-version X.Y.Z] [--software-gl] [--python PATH.py]

Launches Blender GUI on the host display from the pristine-Blender Docker
image, with the production extension ZIP installed into a fresh user
profile. Nothing persists between runs except the GPU shader cache
(build/docker_gui_cache -> \$HOME/.cache), which cuts the flame shader's
first-draw pipeline build from ~70s to well under a second.

  --mode MODE            degrade=A, full=B, private=C (default: degrade)
  --zip PATH             extension ZIP to install. Default resolves from dist/
                         by mode (thyllore_animation_curve_copilot_<mode>-*.zip)
  --scene PATH           .blend file to open (must be inside the repo;
                         default: empty startup scene). The scene is copied
                         to build/docker_gui/ (mounted writable) so saving
                         inside Blender works; the copy is deleted when this
                         script exits
  --blender-version VER  Blender to install in the image (default: $BLENDER_VERSION)
  --software-gl          force Mesa llvmpipe (OpenGL) instead of the NVIDIA GPU
                         (Vulkan backend, the default)
  --no-install           skip the automatic extension install; the ZIP (if
                         given) is still mounted at /zips for manual install
                         via Edit > Preferences > Add-ons > Install from Disk.
                         Without --zip nothing is mounted (plain Blender)
  --python PATH          Python script (inside the repo) run by Blender after
                         the scene is loaded
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --zip) ZIP="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --blender-version) BLENDER_VERSION="$2"; shift 2 ;;
        --software-gl) SOFTWARE_GL=1; shift ;;
        --no-install) NO_INSTALL=1; shift ;;
        --python) PYTHON_SCRIPT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    degrade|full|private) ;;
    *) echo "invalid mode: $MODE (expected degrade, full or private)" >&2; exit 2 ;;
esac
if [[ -z "$ZIP" && "$NO_INSTALL" -eq 0 ]]; then
    ZIP="$REPO_ROOT/dist/thyllore_animation_curve_copilot_${MODE/degrade/degraded}-0.0.1-linux_x86_64.zip"
fi
if [[ -n "$ZIP" && ! -f "$ZIP" ]]; then
    echo "extension ZIP not found: $ZIP (build it with scripts/build_blender_addon.sh)" >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set - run from a graphical session" >&2
    exit 1
fi

docker build --build-arg "BLENDER_VERSION=$BLENDER_VERSION" -t "$IMAGE_TAG" \
    "$REPO_ROOT/blender/docker"

xhost +si:localuser:root >/dev/null

ZIP_MOUNT=()
ZIP_NAME=""
if [[ -n "$ZIP" ]]; then
    ZIP_MOUNT=(-v "$(cd "$(dirname "$ZIP")" && pwd):/zips:ro")
    ZIP_NAME="$(basename "$ZIP")"
fi

GPU_FLAGS=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all)
if [[ -d /dev/dri ]]; then
    GPU_FLAGS+=(--device /dev/dri)
fi
GL_ENV=()
BACKEND_ARG="--gpu-backend vulkan"
if [[ "$SOFTWARE_GL" -eq 1 ]]; then
    GPU_FLAGS=()
    GL_ENV=(-e LIBGL_ALWAYS_SOFTWARE=1)
    BACKEND_ARG="--gpu-backend opengl"
fi

SCREENSHOT_DIR_HOST="/tmp/thyllore_screenshots"
mkdir -p "$SCREENSHOT_DIR_HOST"

GPU_CACHE_DIR="$REPO_ROOT/build/docker_gui_cache"
mkdir -p "$GPU_CACHE_DIR"

SCENE_PREP=""
SCENE_ARG=""
WORK_DIR="$REPO_ROOT/build/docker_gui"
mkdir -p "$WORK_DIR"
if [[ -n "$SCENE" ]]; then
    if ! SCENE_ABS="$(realpath -e "$SCENE" 2>/dev/null)"; then
        echo "scene file not found: $SCENE" >&2
        exit 1
    fi
    if [[ "$SCENE_ABS" != "$REPO_ROOT"/* ]]; then
        echo "scene must be inside the repo (mounted at /workspace): $SCENE" >&2
        exit 1
    fi
    SCENE_ARG="/scenes/$(basename "$SCENE_ABS")"
    SCENE_PREP="cp -f '/workspace${SCENE_ABS#"$REPO_ROOT"}' '$SCENE_ARG' &&"
    SCENE_COPY_HOST="$WORK_DIR/$(basename "$SCENE_ABS")"
    trap 'rm -f "$SCENE_COPY_HOST"; rmdir "$WORK_DIR" 2>/dev/null || true' EXIT
fi

PYTHON_ARG=""
if [[ -n "$PYTHON_SCRIPT" ]]; then
    if ! PYTHON_ABS="$(realpath -e "$PYTHON_SCRIPT" 2>/dev/null)" || [[ "$PYTHON_ABS" != "$REPO_ROOT"/* ]]; then
        echo "python script must exist inside the repo: $PYTHON_SCRIPT" >&2
        exit 1
    fi
    PYTHON_ARG="--python /workspace${PYTHON_ABS#"$REPO_ROOT"}"
fi

docker run --rm \
    ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
    ${GL_ENV[@]+"${GL_ENV[@]}"} \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v "$REPO_ROOT:/workspace:ro" \
    ${ZIP_MOUNT[@]+"${ZIP_MOUNT[@]}"} \
    -v "$WORK_DIR:/scenes" \
    -v "$SCREENSHOT_DIR_HOST:/screenshots" \
    -v "$GPU_CACHE_DIR:/tmp/blender_home/.cache" \
    -e THYLLORE_SCREENSHOT_DIR -e THYLLORE_SCREENSHOT_DELAY -e THYLLORE_SCREENSHOT_QUIT \
    "$IMAGE_TAG" \
    bash -c "
        export HOME=/tmp/blender_home && mkdir -p \"\$HOME\" &&
        if [[ $NO_INSTALL -eq 0 ]]; then
            blender --command extension install-file -r user_default --enable /zips/$ZIP_NAME
        fi &&
        $SCENE_PREP
        exec blender $BACKEND_ARG $SCENE_ARG $PYTHON_ARG
    "
