#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WHEELS_DIR="blender_addon/wheels"
SKIP_MATURIN=0
DEBUG_LOG=0
CRATE="thyllore-ml-core"

usage() {
    cat <<EOF
Usage: $0 [options]

Linux/macOS port of scripts/collect_wheels.ps1. Reproduces the GitHub Actions
"Collect vendored wheels" step locally so failures can be diagnosed without
pushing.

Options:
  --wheels-dir PATH              Output directory (default: blender_addon/wheels)
  --skip-maturin                 Skip building thyllore_ml_core wheel via maturin
  --crate <name>                 Crate to build (default: thyllore-ml-core)
  --debug-log                    Build the wheel with the debug-log feature
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wheels-dir)        WHEELS_DIR="$2"; shift 2 ;;
        --skip-maturin)      SKIP_MATURIN=1; shift ;;
        --debug-log)         DEBUG_LOG=1; shift ;;
        --crate)             CRATE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$CRATE" == "thyllore-effect-core" && "$DEBUG_LOG" -eq 1 ]]; then
    echo "error: --crate thyllore-effect-core cannot be combined with --debug-log" >&2
    echo "       The debug-log feature is specific to thyllore-ml-core." >&2
    exit 1
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

if [[ "$SKIP_MATURIN" -eq 0 ]]; then
    MATURIN_FEATURES="python"
    if [[ "$DEBUG_LOG" -eq 1 ]]; then
        MATURIN_FEATURES="python,debug-log"
        echo "[collect_wheels] Building DEBUG wheel (features: $MATURIN_FEATURES)"
    else
        echo "[collect_wheels] Building $CRATE wheel via maturin..."
    fi
    "$PYTHON_BIN" -m pip install --quiet maturin
    (
        cd "$REPO_ROOT/crates/$CRATE"
        export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$HOME=."
        maturin build --release --features "$MATURIN_FEATURES" --out "$ABS_WHEELS"
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
