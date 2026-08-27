#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLATFORM=""
VERSION="0.0.1"
OUTPUT_DIR="dist"
SKIP_VALIDATE=0
KEEP_STAGE=0

usage() {
    cat <<EOF
Usage: $0 --platform PLATFORM [options]

Build the Blender Flame addon ZIP for the requested platform.

Required:
  --platform PLATFORM            One of: win_amd64, linux_x86_64, macosx_arm64

Options:
  --version VERSION              Extension version (default: $VERSION)
  --output-dir PATH              Output directory (default: $OUTPUT_DIR)
 --skip-validate                Skip "blender --command extension validate"
  --keep-stage                   Keep the temporary stage directory after ZIP creation
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)             PLATFORM="$2"; shift 2 ;;
        --version)              VERSION="$2"; shift 2 ;;
       --skip-validate)        SKIP_VALIDATE=1; shift ;;
        --keep-stage)           KEEP_STAGE=1; shift ;;
        -h|--help)              usage; exit 0 ;;
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

echo "[build_blender_flame_addon] Platform: $PLATFORM -> Blender: $BLENDER_NAME, version: $VERSION"

echo "[build_blender_flame_addon] Exporting flame GLSL shaders..."
python3 scripts/export_flame_glsl.py --repo-root "$REPO_ROOT" --out "$REPO_ROOT/blender_flame_addon/shaders"

if ! ls "$REPO_ROOT/blender_flame_addon/wheels/thyllore_effect_core-"*.whl >/dev/null 2>&1; then
    echo "[build_blender_flame_addon] Collecting wheels..."
    bash scripts/collect_wheels.sh --crate thyllore-effect-core --wheels-dir blender_flame_addon/wheels
fi

STAGE_DIR="$REPO_ROOT/build/blender_flame_addon_stage_${PLATFORM}"
SOURCE_DIR="$REPO_ROOT/blender_flame_addon"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

if command -v rsync >/dev/null 2>&1; then
    rsync -a \
        --exclude='tests/' \
        --exclude='tools/' \
        --exclude='.gitignore' \
        --exclude='__pycache__/' \
        --exclude='.pytest_cache/' \
        --exclude='*.pyc' \
        "$SOURCE_DIR/" "$STAGE_DIR/"
else
    cp -a "$SOURCE_DIR/." "$STAGE_DIR/"
    find "$STAGE_DIR" -type d \( -name tests -o -name tools -o -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
    find "$STAGE_DIR" -type f -name ".gitignore" -delete
    find "$STAGE_DIR" -type f -name "*.pyc" -delete
fi

if [[ ! -f "$STAGE_DIR/shaders/flame_resolve.glsl" ]]; then
    echo "shaders/flame_resolve.glsl missing from stage" >&2
    exit 1
fi
if [[ ! -f "$STAGE_DIR/shaders/flame_resolve.bindings.json" ]]; then
    echo "shaders/flame_resolve.bindings.json missing from stage" >&2
    exit 1
fi
WHEELS_DIR="$STAGE_DIR/wheels"
if [[ ! -d "$WHEELS_DIR" ]]; then
    echo "Stage directory has no wheels/." >&2
    exit 1
fi

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
    echo "Expected at least 1 wheel for $PLATFORM, got ${#KEPT_WHEELS[@]}" >&2
    exit 1
fi
echo "[build_blender_flame_addon] Kept ${#KEPT_WHEELS[@]} wheels for $PLATFORM"

MANIFEST_PATH="$STAGE_DIR/blender_manifest.toml"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "blender_manifest.toml missing from stage" >&2
    exit 1
fi

WHEEL_LINES_JOINED=""
for w in "${KEPT_WHEELS[@]}"; do
    WHEEL_LINES_JOINED+="    \"./wheels/$w\",\n"
done

WHEEL_LINES_JOINED="$WHEEL_LINES_JOINED" \
MANIFEST_PATH="$MANIFEST_PATH" \
BLENDER_NAME="$BLENDER_NAME" \
VERSION="$VERSION" \
python3 - <<'PY'
import os, re
manifest_path = os.environ["MANIFEST_PATH"]
blender_name = os.environ["BLENDER_NAME"]
version = os.environ["VERSION"]
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
text = re.sub(r'version = "0\.0\.1"', f'version = "{version}"', text)

with open(manifest_path, "w", encoding="utf-8", newline="") as f:
    f.write(text)
PY

echo "[build_blender_flame_addon] Manifest updated"

if [[ "$SKIP_VALIDATE" -ne 1 ]]; then
    echo "[build_blender_flame_addon] Validating with Blender..."
    if ! docker image inspect thyllore-blender-xvfb:local >/dev/null 2>&1; then
        echo "[build_blender_flame_addon] Building Blender docker image..."
        docker build -f blender/docker/Dockerfile.xvfb -t thyllore-blender-xvfb:local blender/docker
    fi
  LOG_FILE="$REPO_ROOT/build/extension_validate_${PLATFORM}.log"
    if ! docker run --rm \
        -v "$REPO_ROOT:$REPO_ROOT" \
        -w "$REPO_ROOT" \
        thyllore-blender-xvfb:local \
        sh -c "xvfb-run -a -s '-screen 0 1280x720x24' blender --command extension validate '$STAGE_DIR' > '$LOG_FILE' 2>&1"; then
        cat "$LOG_FILE" >&2
        exit 1
    fi
    tail -5 "$LOG_FILE"
else
    echo "[build_blender_flame_addon] Skipping validation (--skip-validate)"
fi

ABS_OUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
mkdir -p "$ABS_OUT_DIR"

ZIP_PATH="$ABS_OUT_DIR/thyllore_flame-${VERSION}-${PLATFORM}.zip"
rm -f "$ZIP_PATH"

(
    cd "$STAGE_DIR"
    zip -q -r -X "$ZIP_PATH" .
)

SIZE_BYTES="$(stat -c '%s' "$ZIP_PATH" 2>/dev/null || stat -f '%z' "$ZIP_PATH")"

echo
echo "[build_blender_flame_addon] Created: $ZIP_PATH"
echo "[build_blender_flame_addon] Size:    $SIZE_BYTES bytes"

if [[ "$KEEP_STAGE" -ne 1 ]]; then
    rm -rf "$STAGE_DIR"
fi
