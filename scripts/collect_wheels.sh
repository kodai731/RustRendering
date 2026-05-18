#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WHEELS_DIR="blender_addon/wheels"
SKIP_MATURIN=0
SKIP_PIP_DOWNLOAD=0
PLATFORMS=("win_amd64" "manylinux2014_x86_64" "macosx_11_0_arm64")
GRPCIO_VERSION="1.71.2"
PROTOBUF_VERSION="5.29.6"
CERTIFI_VERSION="2024.12.14"
PYTHON_ABI="cp311"
PYTHON_VERSION="3.11"
VARIANT="lite"

PLATFORMS_OVERRIDE=()

usage() {
    cat <<EOF
Usage: $0 [options]

Linux/macOS port of scripts/collect_wheels.ps1. Reproduces the GitHub Actions
"Collect vendored wheels" step locally so failures can be diagnosed without
pushing.

Options:
  --wheels-dir PATH              Output directory (default: blender_addon/wheels)
  --skip-maturin                 Skip building thyllore_ml_core wheel via maturin
  --skip-pip-download            Skip downloading Tier A runtime wheels
  --platforms "p1 p2 ..."        Space-separated platform tags
                                 (default: win_amd64 manylinux2014_x86_64 macosx_11_0_arm64)
  --grpcio-version VERSION       grpcio / grpcio-status version (default: $GRPCIO_VERSION)
  --protobuf-version VERSION     protobuf version (default: $PROTOBUF_VERSION)
  --certifi-version VERSION      certifi version (default: $CERTIFI_VERSION)
  --python-abi TAG               Python ABI tag (default: $PYTHON_ABI)
  --python-version VERSION       Python version (default: $PYTHON_VERSION)
  --variant lite|full            "lite" skips Tier A wheels (default: $VARIANT)
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheels-dir)        WHEELS_DIR="$2"; shift 2 ;;
        --skip-maturin)      SKIP_MATURIN=1; shift ;;
        --skip-pip-download) SKIP_PIP_DOWNLOAD=1; shift ;;
        --platforms)
            read -r -a PLATFORMS_OVERRIDE <<<"$2"
            shift 2
            ;;
        --grpcio-version)    GRPCIO_VERSION="$2"; shift 2 ;;
        --protobuf-version)  PROTOBUF_VERSION="$2"; shift 2 ;;
        --certifi-version)   CERTIFI_VERSION="$2"; shift 2 ;;
        --python-abi)        PYTHON_ABI="$2"; shift 2 ;;
        --python-version)    PYTHON_VERSION="$2"; shift 2 ;;
        --variant)
            case "$2" in
                lite|full) VARIANT="$2" ;;
                *) echo "invalid variant: $2 (expected lite or full)" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ${#PLATFORMS_OVERRIDE[@]} -gt 0 ]]; then
    PLATFORMS=("${PLATFORMS_OVERRIDE[@]}")
fi

HOST_PYTHON="${PYTHON:-python3}"
if ! command -v "$HOST_PYTHON" >/dev/null 2>&1; then
    echo "python3 not found on PATH (set PYTHON=... to override)" >&2
    exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$HOST_PYTHON"
    echo "[collect_wheels] Using active venv: $VIRTUAL_ENV"
else
    VENV_DIR="${THYLLORE_COLLECT_WHEELS_VENV:-$REPO_ROOT/.venv-collect-wheels}"
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        echo "[collect_wheels] Creating venv at $VENV_DIR"
        "$HOST_PYTHON" -m venv "$VENV_DIR"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"
    PATH="$VENV_DIR/bin:$PATH"
    if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
        echo "[collect_wheels] Bootstrapping pip in venv"
        "$PYTHON_BIN" -m ensurepip --upgrade
    fi
fi

ABS_WHEELS="$REPO_ROOT/$WHEELS_DIR"
mkdir -p "$ABS_WHEELS"

if [[ "$SKIP_PIP_DOWNLOAD" -eq 0 ]]; then
    if [[ "$VARIANT" == "lite" ]]; then
        echo "[collect_wheels] Variant=lite -- skipping Tier A wheels (grpcio / protobuf / certifi)"
    else
        echo "[collect_wheels] Downloading vendored runtime wheels..."
        for platform in "${PLATFORMS[@]}"; do
            echo "  Platform: $platform"
            "$PYTHON_BIN" -m pip download \
                "grpcio==$GRPCIO_VERSION" \
                "grpcio-status==$GRPCIO_VERSION" \
                "protobuf==$PROTOBUF_VERSION" \
                "certifi==$CERTIFI_VERSION" \
                --platform "$platform" \
                --abi "$PYTHON_ABI" \
                --implementation cp \
                --python-version "$PYTHON_VERSION" \
                --only-binary=:all: \
                --dest "$ABS_WHEELS" \
                --no-deps
        done
    fi
fi

if [[ "$SKIP_MATURIN" -eq 0 ]]; then
    echo "[collect_wheels] Building thyllore_ml_core wheel via maturin..."
    "$PYTHON_BIN" -m pip install --quiet maturin
    (
        cd "$REPO_ROOT/crates/thyllore-ml-core"
        maturin build --release --features python --out "$ABS_WHEELS"
    )
fi

echo "[collect_wheels] Writing local SHA256 manifest (gitignored)..."
HASHES_FILE="$ABS_WHEELS/HASHES.txt"
: >"$HASHES_FILE"
shopt -s nullglob
mapfile -t wheel_files < <(printf "%s\n" "$ABS_WHEELS"/*.whl | LC_ALL=C sort)
shopt -u nullglob
for wheel in "${wheel_files[@]}"; do
    hash="$(sha256sum "$wheel" | awk '{print $1}')"
    echo "$(basename "$wheel")  $hash" >>"$HASHES_FILE"
done

wheel_count=${#wheel_files[@]}
echo "[collect_wheels] Done. $wheel_count wheels in $ABS_WHEELS"
