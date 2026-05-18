#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "$HOME/.cargo/bin" && ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

BLENDER_VERSION="4.2.0"
if [[ -n "${THYLLORE_BLENDER_INSTALL_DIR:-}" ]]; then
    BLENDER_INSTALL_DIR="$THYLLORE_BLENDER_INSTALL_DIR"
else
    BLENDER_INSTALL_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/thyllore/blender-${BLENDER_VERSION}"
fi
BLENDER_EXTRACTED_DIR="$BLENDER_INSTALL_DIR/blender-${BLENDER_VERSION}-linux-x64"
DEFAULT_BLENDER_EXE="$BLENDER_EXTRACTED_DIR/blender"
WSL_BLENDER_EXE="$HOME/blender_test/blender/blender"
BLENDER_URL="https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

ORT_VERSION="1.23.2"
ORT_VENDOR_DIR="$REPO_ROOT/vendor/onnxruntime/onnxruntime-linux-x64-${ORT_VERSION}"
ORT_DYLIB_PATH_LINUX="$ORT_VENDOR_DIR/lib/libonnxruntime.so"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

FIXTURE_ROOT="$REPO_ROOT/fixtures/ml_parity"
ONNX_REPO="kodai731/thyllore-curve-copilot"
ONNX_REVISION="${ONNX_REVISION:-3c4e8171e0a99a46e7a74b138c12a61a5d444154}"
ONNX_LOCAL_PATH="$FIXTURE_ROOT/onnx/curve_copilot.onnx"
ONNX_DOWNLOAD_URL="https://huggingface.co/${ONNX_REPO}/resolve/${ONNX_REVISION}/curve_copilot.onnx"

CARGO_TARGET_DIR_OVERRIDE="${CARGO_TARGET_DIR:-$REPO_ROOT/target-linux}"

SKIP_FIXTURE_GENERATION=0
SKIP_BUILD=0
TEST_SUBSET="all"
BLENDER_EXE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-fixture-generation) SKIP_FIXTURE_GENERATION=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --blender-exe) BLENDER_EXE_OVERRIDE="$2"; shift 2 ;;
        --test-subset)
            case "$2" in
                all|blender_parity|operator_smoke|wire_parity) TEST_SUBSET="$2" ;;
                *) echo "invalid test subset: $2"; exit 2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--skip-fixture-generation] [--skip-build] [--blender-exe PATH]
          [--test-subset all|blender_parity|operator_smoke|wire_parity]

Reproduces .github/workflows/blender_parity.yml on the local Linux/WSL2 host.
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

write_step() { printf "\n==> %s\n" "$1"; }

resolve_blender_executable() {
    if [[ -n "$BLENDER_EXE_OVERRIDE" && -x "$BLENDER_EXE_OVERRIDE" ]]; then
        echo "$BLENDER_EXE_OVERRIDE"
        return
    fi
    if [[ -n "${THYLLORE_BLENDER_PATH:-}" && -x "$THYLLORE_BLENDER_PATH" ]]; then
        echo "$THYLLORE_BLENDER_PATH"
        return
    fi
    if [[ -x "$WSL_BLENDER_EXE" ]]; then
        echo "$WSL_BLENDER_EXE"
        return
    fi
    if [[ -x "$DEFAULT_BLENDER_EXE" ]]; then
        echo "$DEFAULT_BLENDER_EXE"
        return
    fi
    echo ""
}

install_blender42() {
    write_step "Downloading Blender ${BLENDER_VERSION} to $BLENDER_INSTALL_DIR"
    mkdir -p "$BLENDER_INSTALL_DIR"
    local archive="$BLENDER_INSTALL_DIR/blender.tar.xz"
    curl -L --fail --retry 3 --retry-delay 2 -o "$archive" "$BLENDER_URL"
    tar -xf "$archive" -C "$BLENDER_INSTALL_DIR"
    rm -f "$archive"
}

install_onnx_runtime() {
    write_step "Downloading ONNX Runtime ${ORT_VERSION}"
    mkdir -p "$REPO_ROOT/vendor/onnxruntime"
    local archive="$REPO_ROOT/vendor/onnxruntime/ort.tgz"
    curl -L --fail --retry 3 --retry-delay 2 -o "$archive" "$ORT_URL"
    tar -xzf "$archive" -C "$REPO_ROOT/vendor/onnxruntime"
    rm -f "$archive"
}

download_curve_copilot_onnx() {
    write_step "Downloading curve_copilot.onnx from HuggingFace"
    mkdir -p "$(dirname "$ONNX_LOCAL_PATH")"
    curl -L --fail --retry 3 --retry-delay 2 -o "$ONNX_LOCAL_PATH" "$ONNX_DOWNLOAD_URL"
}

run_cargo_test() {
    local crate="$1" test_file="$2" test_name="${3:-}" features="${4:-}" ignored="${5:-0}"
    local args=(test -p "$crate" --test "$test_file")
    [[ -n "$features" ]] && args+=(--features "$features")
    [[ -n "$test_name" ]] && args+=("$test_name")
    args+=(-- --nocapture)
    [[ "$ignored" == "1" ]] && args+=(--ignored)
    echo "  cargo ${args[*]}"
    CARGO_TARGET_DIR="$CARGO_TARGET_DIR_OVERRIDE" cargo "${args[@]}"
}

write_step "Resolving prerequisites"

if [[ ! -f "$ORT_DYLIB_PATH_LINUX" ]]; then
    install_onnx_runtime
fi
echo "ONNX Runtime: $ORT_DYLIB_PATH_LINUX"

resolved_blender="$(resolve_blender_executable)"
if [[ -z "$resolved_blender" ]]; then
    install_blender42
    resolved_blender="$DEFAULT_BLENDER_EXE"
fi
if [[ ! -x "$resolved_blender" ]]; then
    echo "Blender executable not found at $resolved_blender" >&2
    exit 1
fi
echo "Blender: $resolved_blender"
"$resolved_blender" --version | head -n 1

if [[ ! -f "$ONNX_LOCAL_PATH" ]]; then
    download_curve_copilot_onnx
fi
echo "curve_copilot.onnx: $ONNX_LOCAL_PATH"

# Phase 5.5 MVP: license verification removed -- the public key backup/restore
# and dev keypair generation steps below are kept commented out so Phase 6
# (Auth Backend reintroduction) can re-enable them in one block.
#
# PUBLIC_KEY_PATH="$REPO_ROOT/blender_addon/license/public_key.pem"
# PUBLIC_KEY_BACKUP=""
# if [[ -f "$PUBLIC_KEY_PATH" ]]; then
#     PUBLIC_KEY_BACKUP="$(mktemp)"
#     cp "$PUBLIC_KEY_PATH" "$PUBLIC_KEY_BACKUP"
# fi
#
# restore_public_key() {
#     if [[ -n "$PUBLIC_KEY_BACKUP" && -f "$PUBLIC_KEY_BACKUP" ]]; then
#         if ! cmp -s "$PUBLIC_KEY_BACKUP" "$PUBLIC_KEY_PATH"; then
#             cp "$PUBLIC_KEY_BACKUP" "$PUBLIC_KEY_PATH"
#             echo "Restored $PUBLIC_KEY_PATH to original"
#         fi
#         rm -f "$PUBLIC_KEY_BACKUP"
#     fi
# }
# trap restore_public_key EXIT

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    # Phase 5.5 MVP: license keypair generation removed (no license module).
    # Phase 6 will reintroduce when the Auth Backend public key is wired back
    # into the addon.
    # write_step "Generating dev keypair (will be restored after run)"
    # pwsh -NoProfile -ExecutionPolicy Bypass -File "$REPO_ROOT/scripts/gen_license_keypair.ps1" || {
    #     echo "(pwsh not available on Linux — skipping keypair, tests use bypass env)"
    # }

    write_step "Collecting vendored wheels (manylinux2014_x86_64, full Variant for Tier A)"
    "$REPO_ROOT/scripts/collect_wheels.sh" \
        --platforms "manylinux2014_x86_64" --variant full

    write_step "Building Blender extension ZIP (linux_x86_64, full Variant)"
    "$REPO_ROOT/scripts/build_blender_addon.sh" \
        --platform linux_x86_64 --variant full --skip-blender-validate
fi

write_step "Installing Blender extension into user dir"
sentinel="$REPO_ROOT/dist/.last_built_zip"
zip_path="$(cat "$sentinel" 2>/dev/null | sed $'1s/^\xEF\xBB\xBF//' || true)"
if [[ -z "$zip_path" || ! -f "$zip_path" ]]; then
    echo "No extension ZIP found via $sentinel. Re-run without --skip-build." >&2
    exit 1
fi

existing_dir="$HOME/.config/blender/4.2/extensions/user_default/thyllore_animation"
if [[ -d "$existing_dir" ]]; then
    echo "Removing existing extension at $existing_dir"
    rm -rf "$existing_dir"
fi

echo "Installing $zip_path"
"$resolved_blender" --command extension install-file --repo user_default --enable "$zip_path"

export THYLLORE_PARITY_FIXTURE_OUTPUT="$FIXTURE_ROOT"
export THYLLORE_BLENDER_PATH="$resolved_blender"
# Phase 5.5 MVP: THYLLORE_TEST_BYPASS_LICENSE removed (no license module).
# Phase 6 will reintroduce when Auth Backend ships.
# export THYLLORE_TEST_BYPASS_LICENSE=1
export THYLLORE_HEADLESS=1
export ORT_DYLIB_PATH="$ORT_DYLIB_PATH_LINUX"

if [[ "$SKIP_FIXTURE_GENERATION" -eq 0 ]]; then
    write_step "Generating proto/numpy parity fixtures"
    run_cargo_test thyllore-grpc-client grpc_fixture_generator \
        generate_grpc_request_response_fixtures "auto-rig,text-to-motion" 1
    run_cargo_test thyllore-ml-core curve_copilot_fixture_generator \
        generate_curve_copilot_input_and_golden_fixtures "" 1
fi

run_blender_parity=0
run_operator_smoke=0
run_wire_parity=0
case "$TEST_SUBSET" in
    all)             run_blender_parity=1; run_operator_smoke=1; run_wire_parity=1 ;;
    blender_parity)  run_blender_parity=1 ;;
    operator_smoke)  run_operator_smoke=1 ;;
    wire_parity)     run_wire_parity=1 ;;
esac

if [[ "$run_blender_parity" -eq 1 ]]; then
    write_step "Running curve_copilot_blender_parity (typed vs call_op_json)"
    run_cargo_test thyllore-ml-core curve_copilot_blender_parity \
        curve_copilot_typed_and_call_op_paths_match_in_blender "" 1
fi

if [[ "$run_operator_smoke" -eq 1 ]]; then
    write_step "Running curve_copilot_operator_smoke"
    run_cargo_test thyllore-grpc-client curve_copilot_operator_smoke \
        curve_copilot_operator_runs_under_background "auto-rig,text-to-motion" 1
fi

if [[ "$run_wire_parity" -eq 1 ]]; then
    write_step "Running grpc_request_wire_parity"
    run_cargo_test thyllore-grpc-client grpc_request_wire_parity \
        grpc_request_wire_bytes_match_across_rust_and_blender "auto-rig,text-to-motion" 1
fi

write_step "All requested parity tests passed locally"
