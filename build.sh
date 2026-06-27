#!/usr/bin/env bash
# Linux build entry point. Ensures the patched openusd is vendored (idempotent)
# before invoking cargo, so `[patch.crates-io] openusd = vendor/openusd` always
# resolves. Mirrors build-with-tests.ps1's vendor step for Windows.
#
# Usage: ./build.sh <cargo-args>
#   ./build.sh build
#   ./build.sh build --release
#   ./build.sh test --lib
set -euo pipefail
cd "$(dirname "$0")"

./scripts/vendor_openusd_patch.sh

exec cargo "$@"
