ORT_VERSION="1.23.2"
ORT_VENDOR_DIR="$REPO_ROOT/vendor/onnxruntime"
ORT_DYLIB="$ORT_VENDOR_DIR/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

ensure_onnxruntime() {
    if [[ -z "${REPO_ROOT:-}" ]]; then
        echo "error: REPO_ROOT is not set" >&2
        return 1
    fi
    if [[ -f "$ORT_DYLIB" ]]; then
        printf "\n==> ONNX Runtime present: %s\n" "$ORT_DYLIB"
        return
    fi
    printf "\n==> Downloading ONNX Runtime %s\n" "${ORT_VERSION}"
    mkdir -p "$ORT_VENDOR_DIR"
    local archive="$ORT_VENDOR_DIR/ort.tgz"
    curl -L --fail --retry 3 --retry-delay 2 -o "$archive" "$ORT_URL"
    tar -xzf "$archive" -C "$ORT_VENDOR_DIR"
    rm -f "$archive"
}
