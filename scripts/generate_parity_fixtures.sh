#!/usr/bin/env bash
set -euo pipefail

FORCE=0
SHARED_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --shared-data-path) SHARED_DATA_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--force] [--shared-data-path PATH]"
            echo "Regenerates the ml_parity fixture set from cargo test output and"
            echo "refreshes manifest.json under <SharedDataPath>/fixtures/ml_parity/."
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

EXPORTS_DIR="$SHARED_DATA_PATH/exports"
if [[ ! -d "$EXPORTS_DIR" ]]; then
    echo "ERROR: $EXPORTS_DIR not found" >&2
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

echo "==> generating Tier A proto fixtures"
(
    cd "$WORKSPACE_ROOT"
    CARGO_TARGET_DIR="$WORKSPACE_ROOT/target-linux" \
        cargo test -p thyllore-grpc-client --features auto-rig,text-to-motion \
        --test parity_fixtures_phase5 generate_phase5_proto_fixtures \
        -- --ignored --nocapture
)

echo "==> generating Tier B (curve_copilot) input + golden fixtures"
# WSL2 has no Linux onnxruntime.so vendored; delegate Tier B inference to the
# Windows host cargo, which has the vendored onnxruntime.dll.
if [[ -e /proc/version && $(grep -ci microsoft /proc/version) -gt 0 ]]; then
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

COMMIT=$(cd "$WORKSPACE_ROOT" && git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
GENERATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "==> writing manifest.json"
python3 - "$FIXTURE_ROOT" "$COMMIT" "$GENERATED_AT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

fixture_root = Path(sys.argv[1])
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

excluded_filenames = {"manifest.json", "README.md", ".gitkeep"}
for path in sorted(fixture_root.rglob("*")):
    if not path.is_file() or path.name in excluded_filenames:
        continue
    rel = str(path.relative_to(fixture_root)).replace(os.sep, "/")
    data = path.read_bytes()
    manifest["fixtures"][rel] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

(fixture_root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(f"manifest written: {fixture_root}/manifest.json ({len(manifest['fixtures'])} entries)")
PY

echo
echo "fixtures regenerated at $FIXTURE_ROOT"
echo "next: commit manifest.json + new files in SharedData repository:"
echo "  cd $SHARED_DATA_PATH"
echo "  git status fixtures/ml_parity"
echo "  git add fixtures/ml_parity"
echo "  git commit -m \"regenerate ml_parity fixtures (ThylloreAnimation @ $COMMIT)\""
