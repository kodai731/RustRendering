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
LOG="$REPO_ROOT/log/blender_flame_probe/addon_smoke.log"
docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" thyllore-blender-xvfb:local sh -c "xvfb-run -a -s '-screen 0 1280x720x24' blender -noaudio --python-exit-code 1 --python '$REPO_ROOT/blender_flame_addon/tests/addon_smoke.py' > '$LOG' 2>&1" || true

grep -v '^\s*|' "$LOG" | grep -v gpu.debug | tail -20

grep -q "^ADDON_SMOKE ok" "$LOG" && grep -q "^DRAW_SMOKE ok" "$LOG"
