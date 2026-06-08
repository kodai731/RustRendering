#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "$HOME/.cargo/bin" && ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

ORT_VERSION="1.23.2"
ORT_VENDOR_DIR="$REPO_ROOT/vendor/onnxruntime"
ORT_DYLIB="$ORT_VENDOR_DIR/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

HF_REPO="kodai731/thyllore-curve-copilot"
HF_REVISION="${THYLLORE_HF_REVISION:-main}"
HF_MODEL_FILENAME="curve_copilot.onnx"
HF_MODEL_URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/${HF_MODEL_FILENAME}"
MODEL_DIR="$REPO_ROOT/target/ci-models"
MODEL_PATH="$MODEL_DIR/${HF_MODEL_FILENAME}"

write_step() { printf "\n==> %s\n" "$1"; }

ensure_onnxruntime() {
    if [[ -f "$ORT_DYLIB" ]]; then
        write_step "ONNX Runtime present: $ORT_DYLIB"
        return
    fi
    write_step "Downloading ONNX Runtime ${ORT_VERSION}"
    mkdir -p "$ORT_VENDOR_DIR"
    local archive="$ORT_VENDOR_DIR/ort.tgz"
    curl -L --fail --retry 3 --retry-delay 2 -o "$archive" "$ORT_URL"
    tar -xzf "$archive" -C "$ORT_VENDOR_DIR"
    rm -f "$archive"
}

download_model() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "HF_TOKEN is required: ${HF_REPO} is a private HuggingFace repo." >&2
        echo "Set the HF_TOKEN environment variable (read-scoped access token)." >&2
        exit 1
    fi
    write_step "Downloading v1 curve_copilot model from private HuggingFace (${HF_REPO}@${HF_REVISION})"
    mkdir -p "$MODEL_DIR"
    curl -L --fail --retry 3 --retry-delay 2 \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        -o "$MODEL_PATH" "$HF_MODEL_URL"
}

ensure_onnxruntime
download_model

write_step "Running v1 curve_copilot inference smoke test against the model"
export THYLLORE_CURVE_MODEL="$MODEL_PATH"
cargo test -p thyllore-ml-core --test v1_curve_copilot_inference_smoke -- --ignored --nocapture
