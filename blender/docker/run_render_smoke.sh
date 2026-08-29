#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! docker image inspect thyllore-blender-xvfb:local >/dev/null 2>&1; then
    docker build -f blender/docker/Dockerfile.xvfb -t thyllore-blender-xvfb:local blender/docker
fi

if ! ls blender_flame_addon/wheels/thyllore_effect_core-*.whl >/dev/null 2>&1; then
    (cd crates/thyllore-effect-core && uvx maturin build --release --features python --out "$REPO_ROOT/blender_flame_addon/wheels")
fi

mkdir -p log/blender_flame_probe
LOG="$REPO_ROOT/log/blender_flame_probe/render_smoke.log"
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" thyllore-blender-xvfb:local sh -c "xvfb-run -a -s '-screen 0 1280x720x24' blender --gpu-backend vulkan -noaudio --python-exit-code 1 --python '$REPO_ROOT/blender_flame_addon/tests/render_smoke.py' > '$LOG' 2>&1" || true

grep -v '^\s*|' "$LOG" | grep -v gpu.debug | tail -20

grep -q "^RENDER_SMOKE ok" "$LOG" && grep -q "^COMPOSITOR_SMOKE ok" "$LOG" && grep -qE "^DEPTH_SMOKE (ok|skipped)" "$LOG"
