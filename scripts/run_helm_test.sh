#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/lib/onnxruntime.sh"
MAIN_ROOT="$(cd "$(dirname "$(git rev-parse --git-common-dir)")" && pwd)"
# Bundle name SSoT: src/ecs/resource/helm_state.rs EXPORTS_BUNDLE_DIR
EXPORTS_BUNDLE_DIR="helm_router_20260821"

# ORT resolution
if [[ -f "$ORT_DYLIB" ]]; then
    :
elif [[ -f "$MAIN_ROOT/vendor/onnxruntime/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so" ]]; then
    export ORT_DYLIB="$MAIN_ROOT/vendor/onnxruntime/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"
else
    ensure_onnxruntime
fi
export ORT_DYLIB_PATH="$ORT_DYLIB"

# Shared Data Dir
if [[ -z "${THYLLORE_SHARED_DATA_DIR:-}" ]] && [[ -d "$MAIN_ROOT/../SharedData" ]]; then
   export THYLLORE_SHARED_DATA_DIR="$MAIN_ROOT/../SharedData"
fi

# Router Model Dir (bundle name SSoT: src/ecs/resource/helm_state.rs EXPORTS_BUNDLE_DIR)
if [[ -z "${THYLLORE_ROUTER_MODEL_DIR:-}" ]]; then
    if [[ -n "${THYLLORE_SHARED_DATA_DIR:-}" ]] && [[ -d "${THYLLORE_SHARED_DATA_DIR}/exports/${EXPORTS_BUNDLE_DIR}/setfit-3ep-camdir" ]]; then
        export THYLLORE_ROUTER_MODEL_DIR="${THYLLORE_SHARED_DATA_DIR}/exports/${EXPORTS_BUNDLE_DIR}/setfit-3ep-camdir"
    else
        export THYLLORE_ROUTER_MODEL_DIR="$REPO_ROOT/models/gemma/setfit-3ep-camdir"
    fi
fi

if [[ ! -f "$THYLLORE_ROUTER_MODEL_DIR/router_index.json" ]]; then
    echo "Error: $THYLLORE_ROUTER_MODEL_DIR/router_index.json is missing." >&2
    echo "Please regenerate it using: AnimationModelTraining scripts/helm_router/export_router_index.py -> promote_to_exports.py" >&2
    exit 2
fi

echo "[helm-test] model dir: $THYLLORE_ROUTER_MODEL_DIR"

# THYLLORE_HELM_HELDOUT: export from default path if missing
if [[ -z "${THYLLORE_HELM_HELDOUT:-}" ]] && [[ -f "$MAIN_ROOT/../AnimationModelTraining/scripts/helm_router/heldout.jsonl" ]]; then
    export THYLLORE_HELM_HELDOUT="$MAIN_ROOT/../AnimationModelTraining/scripts/helm_router/heldout.jsonl"
fi

# Positional arg: layer filter parity|e2e|heldout|all|bench (default all)
LAYER="${1:-all}"
case "$LAYER" in
    parity)   shift || true; TEST_FLAGS="--test helm_router_parity" ;;
    e2e)      shift || true; TEST_FLAGS="--test helm_e2e" ;;
    heldout)  shift || true; TEST_FLAGS="--test helm_heldout" ;;
    all)      shift || true; TEST_FLAGS="--test helm_router_parity --test helm_e2e --test helm_heldout" ;;
    bench)
        shift || true
        echo "[helm-test] building..."
        cargo build
        # NOTE: the batch bench runs the GUI app (headless) — an X display is required.
        exec ./target/debug/thyllore-animation --batch-utterance tests/data/helm_batch_smoke.jsonl --batch-utterance-out log/helm_bench/batch_results.jsonl "$@"
        ;;
    *)        echo "Unknown layer: $LAYER (use parity|e2e|heldout|all|bench)" >&2; exit 1 ;;
esac

exec cargo test $TEST_FLAGS "$@"
