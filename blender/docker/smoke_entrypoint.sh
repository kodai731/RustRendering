#!/usr/bin/env bash
set -euo pipefail

ZIP="$1"
EXPECT_MODE="$2"
RESULT="$3"
shift 3

export HOME="/tmp/blender_home_${EXPECT_MODE}"
rm -rf "$HOME"
mkdir -p "$HOME"

blender --command extension validate "$ZIP"
blender --command extension install-file -r user_default --enable "$ZIP"
blender --background \
    --python /workspace/blender_addon/tests/build_mode_boundary_smoke.py -- \
    --result "$RESULT" --expect-mode "$EXPECT_MODE" "$@"
