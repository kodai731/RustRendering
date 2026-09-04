#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_DIR="$REPO_ROOT/blender_addon/effects/water"
COMMON_DIR="$REPO_ROOT/blender_addon/common"

PLATFORM=""
BUILD_MODE="release"
VERSION="0.0.1"
OUTPUT_DIR="dist"
SKIP_VALIDATE=0
KEEP_STAGE=0

usage() {
    cat <<USAGE
Usage: $0 --platform PLATFORM [options]

Build the Thyllore Water extension ZIP for the requested platform.

Required:
  --platform PLATFORM   One of: win_amd64, linux_x86_64, macosx_arm64

Options:
  --build-mode MODE     release (default) strips the debug/ package;
                        debug keeps it and names the ZIP thyllore_water_debug-*
  --version VERSION     Extension version (default: $VERSION)
  --output-dir PATH     Output directory relative to the repo (default: $OUTPUT_DIR)
  --skip-validate       Skip "blender --command extension validate" (docker)
  --keep-stage          Keep the stage directory after ZIP creation
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)      PLATFORM="$2"; shift 2 ;;
        --build-mode)    BUILD_MODE="$2"; shift 2 ;;
        --version)       VERSION="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --skip-validate) SKIP_VALIDATE=1; shift ;;
        --keep-stage)    KEEP_STAGE=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    echo "--platform is required" >&2
    usage >&2
    exit 2
fi

case "$BUILD_MODE" in
    release) ZIP_BASENAME="thyllore_water" ;;
    debug)   ZIP_BASENAME="thyllore_water_debug" ;;
    *) echo "invalid build mode: $BUILD_MODE (expected release or debug)" >&2; exit 2 ;;
esac

case "$PLATFORM" in
    win_amd64)
        BLENDER_NAME="windows-x64"
        WHEEL_MATCHERS=("win_amd64\\.whl$")
        ;;
    linux_x86_64)
        BLENDER_NAME="linux-x64"
        WHEEL_MATCHERS=(
            "manylinux2014_x86_64\\.whl$"
            "manylinux_2_[0-9]+_x86_64\\.whl$"
            "linux_x86_64\\.whl$"
        )
        ;;
    macosx_arm64)
        BLENDER_NAME="macos-arm64"
        WHEEL_MATCHERS=(
            "macosx_[0-9]+_[0-9]+_arm64\\.whl$"
            "macosx_[0-9]+_[0-9]+_universal2\\.whl$"
        )
        ;;
    *)
        echo "invalid platform: $PLATFORM" >&2
        exit 2
        ;;
esac

log() { echo "[water/build_addon] $*"; }

log "Platform: $PLATFORM -> Blender: $BLENDER_NAME, mode: $BUILD_MODE, version: $VERSION"

log "Exporting water GLSL shaders..."
python3 "$REPO_ROOT/scripts/blender/water/export_glsl.py" --repo-root "$REPO_ROOT" --out "$SOURCE_DIR/shaders"

if ! ls "$SOURCE_DIR/wheels/thyllore_effect_core-"*.whl >/dev/null 2>&1; then
    log "Collecting wheels..."
    bash "$REPO_ROOT/scripts/collect_wheels.sh" --crate thyllore-effect-core --wheels-dir blender_addon/effects/water/wheels
fi

STAGE_DIR="$REPO_ROOT/build/blender_water_addon_stage_${PLATFORM}_${BUILD_MODE}"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

STAGE_EXCLUDES=(tests tools __pycache__ .pytest_cache wheels-extracted)
if [[ "$BUILD_MODE" == "release" ]]; then
    STAGE_EXCLUDES+=(debug)
fi

copy_tree() {
    local source="$1"
    local destination="$2"
    shift 2
    mkdir -p "$destination"
    tar -c "$@" -C "$source" . | tar -x -C "$destination"
}

stage_sources() {
    local exclude_args=(--exclude='.gitignore' --exclude='*.pyc')
    local name
    for name in "${STAGE_EXCLUDES[@]}"; do
        exclude_args+=(--exclude="$name")
    done
    copy_tree "$SOURCE_DIR" "$STAGE_DIR" "${exclude_args[@]}"
    copy_tree "$COMMON_DIR" "$STAGE_DIR/common" --exclude='__pycache__' --exclude='*.pyc'
}
stage_sources

for required in shaders/water_torus.glsl shaders/water_torus.bindings.json common/coordinates.py common/effect_properties.py blender_manifest.toml; do
    if [[ ! -f "$STAGE_DIR/$required" ]]; then
        echo "$required missing from stage" >&2
        exit 1
    fi
done
if [[ "$BUILD_MODE" == "release" && -d "$STAGE_DIR/debug" ]]; then
    echo "debug/ must not be staged for a release build" >&2
    exit 1
fi

WHEELS_DIR="$STAGE_DIR/wheels"
wheel_matches_platform() {
    local name="$1"
    for pattern in "${WHEEL_MATCHERS[@]}"; do
        if [[ "$name" =~ $pattern ]]; then
            return 0
        fi
    done
    return 1
}

KEPT_WHEELS=()
shopt -s nullglob
for wheel_path in "$WHEELS_DIR"/*.whl; do
    name="$(basename "$wheel_path")"
    if wheel_matches_platform "$name"; then
        KEPT_WHEELS+=("$name")
    else
        rm -f "$wheel_path"
    fi
done
shopt -u nullglob
rm -f "$WHEELS_DIR/HASHES.txt" "$WHEELS_DIR/README.md"

if [[ ${#KEPT_WHEELS[@]} -lt 1 ]]; then
    echo "Expected at least 1 wheel for $PLATFORM in $SOURCE_DIR/wheels" >&2
    exit 1
fi
log "Kept ${#KEPT_WHEELS[@]} wheels for $PLATFORM"

WHEEL_LINES_JOINED=""
for w in "${KEPT_WHEELS[@]}"; do
    WHEEL_LINES_JOINED+="    \"./wheels/$w\",\n"
done

WHEEL_LINES_JOINED="$WHEEL_LINES_JOINED" \
MANIFEST_PATH="$STAGE_DIR/blender_manifest.toml" \
BLENDER_NAME="$BLENDER_NAME" \
VERSION="$VERSION" \
python3 - <<'PY'
import os, re
manifest_path = os.environ["MANIFEST_PATH"]
wheel_lines = os.environ["WHEEL_LINES_JOINED"].replace("\\n", "\n").rstrip("\n")
text = open(manifest_path, "rb").read().decode("utf-8-sig")
text = text.replace("PLATFORM_BLENDER_NAME", os.environ["BLENDER_NAME"])
text = re.sub(r"wheels = \[.*?\]", f"wheels = [\n{wheel_lines}\n]", text, flags=re.DOTALL)
text = re.sub(r'version = "0\.0\.1"', f'version = "{os.environ["VERSION"]}"', text)
with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    f.write(text)
PY
log "Manifest updated"

if [[ "$SKIP_VALIDATE" -ne 1 ]]; then
    log "Validating with Blender (docker)..."
    if ! docker image inspect thyllore-blender-xvfb:local >/dev/null 2>&1; then
        docker build -f "$REPO_ROOT/blender/docker/Dockerfile.xvfb" -t thyllore-blender-xvfb:local "$REPO_ROOT/blender/docker"
    fi
    if ! VALIDATE_OUTPUT=$(docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" thyllore-blender-xvfb:local \
        sh -c "xvfb-run -a -s '-screen 0 1280x720x24' blender --command extension validate '$STAGE_DIR' 2>&1"); then
        echo "$VALIDATE_OUTPUT" >&2
        exit 1
    fi
    echo "$VALIDATE_OUTPUT" | tail -5
else
    log "Skipping validation (--skip-validate)"
fi

ABS_OUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
mkdir -p "$ABS_OUT_DIR"
ZIP_PATH="$ABS_OUT_DIR/${ZIP_BASENAME}-${VERSION}-${PLATFORM}.zip"
rm -f "$ZIP_PATH"
(
    cd "$STAGE_DIR"
    zip -q -r -X "$ZIP_PATH" .
)

log "Created: $ZIP_PATH ($(stat -c '%s' "$ZIP_PATH") bytes)"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "zip_path=$ZIP_PATH" >> "$GITHUB_OUTPUT"
fi

if [[ "$KEEP_STAGE" -ne 1 ]]; then
    rm -rf "$STAGE_DIR"
fi
