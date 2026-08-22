#!/usr/bin/env bash
# Preflight check for helm model artifacts.
# Called from scripts/run_engine.sh before cargo run; never stops the engine.

helm_preflight() {
    # Bundle name SSoT: src/ecs/resource/helm_state.rs EXPORTS_BUNDLE_DIR
    local exports_bundle_dir="helm_router_20260821"

    # Resolve model_dir
    local model_dir
    if [[ -n "${THYLLORE_SHARED_DATA_DIR:-}" ]] && \
       [[ -d "${THYLLORE_SHARED_DATA_DIR}/exports/${exports_bundle_dir}/setfit-3ep-camdir" ]]; then
        model_dir="${THYLLORE_SHARED_DATA_DIR}/exports/${exports_bundle_dir}/setfit-3ep-camdir"
    else
        model_dir="$REPO_ROOT/models/gemma/setfit-3ep-camdir"
    fi

    # Resolve raw_dir
    local raw_dir
    if [[ -d "$(dirname "$model_dir")/e5-raw" ]]; then
        raw_dir="$(dirname "$model_dir")/e5-raw"
    else
        raw_dir="$REPO_ROOT/models/gemma/e5-raw"
    fi

    # Check required files
    local missing=()
    for f in router_index.json router_index.f32 raw_index.json raw_index.f32 tokenizer.json onnx/model.onnx; do
        if [[ ! -f "$model_dir/$f" ]]; then
            missing+=("$model_dir/$f")
        fi
    done
    if [[ ! -d "$raw_dir" ]]; then
        missing+=("$raw_dir")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[helm] command bar disabled: missing artifacts:" >&2
        for m in "${missing[@]}"; do
            echo "  $m" >&2
        done
        echo "Regenerate with: AnimationModelTraining scripts/helm_router/export_router_index.py -> promote_to_exports.py" >&2
        echo "(or set THYLLORE_SHARED_DATA_DIR to point at a populated SharedData)." >&2
        echo "engine starts anyway; the command bar will report Unavailable." >&2
    fi

    return 0
}
