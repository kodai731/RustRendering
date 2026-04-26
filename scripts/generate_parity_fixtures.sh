#!/usr/bin/env bash
#
# Phase 5 — Regenerate ml_parity fixtures (recommended path: WSL2 bash).
#
# Usage:
#   bash scripts/generate_parity_fixtures.sh [--force] [--shared-data-path PATH]
#
# Resolves SharedDataPathWSL from .claude/local/paths.md by default and writes
# fixtures into <SharedDataPath>/fixtures/ml_parity/. Calls cargo test to drive
# the per-crate generators (parity_fixtures_phase5.rs in ml-core and
# grpc-client). Refreshes manifest.json with up-to-date SHA-256 sums via
# python3.
#
# This script is read-only on the workspace; it only writes to the SharedData
# fixtures directory and prints a reminder to commit there.

set -euo pipefail

FORCE=0
SHARED_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --shared-data-path) SHARED_DATA_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,17p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATHS_FILE="$WORKSPACE_ROOT/.claude/local/paths.md"

if [[ -z "$SHARED_DATA_PATH" ]]; then
    if [[ ! -f "$PATHS_FILE" ]]; then
        echo "ERROR: paths.md not found at $PATHS_FILE" >&2
        exit 1
    fi
    # Prefer SharedDataPathWSL on Linux/WSL2 for native ext4 I/O.
    SHARED_DATA_PATH=$(grep -E '^- SharedDataPathWSL\s*=' "$PATHS_FILE" \
        | sed -E 's/^- SharedDataPathWSL\s*=\s*//' | tr -d '\r' || true)
    if [[ -z "$SHARED_DATA_PATH" ]]; then
        echo "ERROR: SharedDataPathWSL not found in $PATHS_FILE" >&2
        exit 1
    fi
fi

FIXTURE_ROOT="$SHARED_DATA_PATH/fixtures/ml_parity"
echo "fixture root: $FIXTURE_ROOT"
mkdir -p "$FIXTURE_ROOT/glb" "$FIXTURE_ROOT/proto" "$FIXTURE_ROOT/onnx" "$FIXTURE_ROOT/numpy"

# Refresh canonical onnx from exports/ (latest dated curve_copilot_*.onnx).
EXPORTS_DIR="$SHARED_DATA_PATH/exports"
if [[ ! -d "$EXPORTS_DIR" ]]; then
    echo "ERROR: $EXPORTS_DIR not found; cannot copy curve_copilot.onnx" >&2
    exit 1
fi

LATEST_ONNX=$(ls -1 "$EXPORTS_DIR"/curve_copilot_*.onnx 2>/dev/null | sort | tail -n 1 || true)
if [[ -z "$LATEST_ONNX" ]]; then
    echo "ERROR: no curve_copilot_*.onnx found in $EXPORTS_DIR" >&2
    exit 1
fi
echo "copying onnx: $LATEST_ONNX -> $FIXTURE_ROOT/onnx/curve_copilot.onnx"
cp -f "$LATEST_ONNX" "$FIXTURE_ROOT/onnx/curve_copilot.onnx"

export THYLLORE_PHASE5_FIXTURE_OUTPUT="$FIXTURE_ROOT"

# Tier A (proto fixtures) does not need ORT and can run on Linux/WSL2 native.
# Use a Linux-specific target dir so artifacts don't collide with the Windows
# host's `target/` directory (different rustc, incompatible rlibs).
echo "==> generating Tier A proto fixtures"
(
    cd "$WORKSPACE_ROOT"
    CARGO_TARGET_DIR="$WORKSPACE_ROOT/target-linux" \
        cargo test -p thyllore-grpc-client --features auto-rig,text-to-motion \
        --test parity_fixtures_phase5 generate_phase5_proto_fixtures \
        -- --ignored --nocapture
)

# Tier B (curve_copilot) requires ONNX Runtime. The vendored DLL is
# Windows-only; on WSL2 we delegate to the Windows host cargo.exe via cmd.exe
# so no Linux .so installation is required.
echo "==> generating Tier B (curve_copilot) input + golden fixtures"
if [[ -e /proc/version && $(grep -ci microsoft /proc/version) -gt 0 ]]; then
    # Inside WSL: invoke the Windows cargo. Path conversion ensures the env
    # var arrives in Windows-compatible UNC form.
    WIN_FIXTURE_ROOT=$(echo "$FIXTURE_ROOT" \
        | sed -E 's|^/home/kodai/Projects/SharedData|//wsl.localhost/Ubuntu/home/kodai/Projects/SharedData|')
    WIN_WORKSPACE=$(wslpath -w "$WORKSPACE_ROOT")
    cmd.exe /c "set THYLLORE_PHASE5_FIXTURE_OUTPUT=${WIN_FIXTURE_ROOT}&& cd /D ${WIN_WORKSPACE}&& cargo test -p thyllore-ml-core --test parity_fixtures_phase5 generate_phase5_curve_copilot_fixtures -- --ignored --nocapture" \
        || { echo "ERROR: Windows cargo Tier B generation failed" >&2; exit 1; }
else
    (
        cd "$WORKSPACE_ROOT"
        cargo test -p thyllore-ml-core --test parity_fixtures_phase5 \
            generate_phase5_curve_copilot_fixtures -- --ignored --nocapture
    )
fi

unset THYLLORE_PHASE5_FIXTURE_OUTPUT

# manifest.json — emit via python3 (always available on Ubuntu/WSL2 default)
COMMIT=$(cd "$WORKSPACE_ROOT" && git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
GENERATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "==> writing manifest.json"
python3 - "$FIXTURE_ROOT" "$COMMIT" "$GENERATED_AT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
commit = sys.argv[2]
generated_at = sys.argv[3]

manifest = {
    "schema_version": 1,
    "generated_at": generated_at,
    "generator": "scripts/generate_parity_fixtures.sh",
    "thyllore_animation_commit": commit,
    "proto_version": "v1",
    "fixtures": {},
}

skip_names = {"manifest.json", "README.md", ".gitkeep"}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in skip_names:
        continue
    rel = str(path.relative_to(root)).replace(os.sep, "/")
    data = path.read_bytes()
    manifest["fixtures"][rel] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(f"manifest written: {root}/manifest.json ({len(manifest['fixtures'])} entries)")
PY

echo
echo "fixtures regenerated at $FIXTURE_ROOT"
echo "next: commit manifest.json + new files in SharedData repository:"
echo "  cd $SHARED_DATA_PATH"
echo "  git status fixtures/ml_parity"
echo "  git add fixtures/ml_parity"
echo "  git commit -m \"regenerate ml_parity fixtures (ThylloreAnimation @ $COMMIT)\""
