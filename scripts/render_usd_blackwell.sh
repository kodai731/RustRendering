#!/usr/bin/env bash
# Run the Thyllore USD renderer on the NVIDIA Blackwell GPU inside a CUDA+Vulkan
# container. The container sees only the Blackwell (via CDI), so the engine's
# Vulkan device selection picks it automatically — no engine flag needed.
#
# Usage:
#   scripts/render_usd_blackwell.sh --render-usd <usd> --out <png> --gpu \
#       --camera-pos x,y,z --camera-target x,y,z --fov 39.6 --resolution 512
#
# With no arguments it opens an interactive shell in the container.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_root}/docker/gpu-runtime.env"

image=thyllore-render-cuda:latest
if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Image ${image} not found. Build it first:" >&2
    echo "  docker compose -f docker/docker-compose.cuda.yaml --profile cuda build" >&2
    exit 1
fi

run_args=(
    --rm --init
    --device "${CDI_GPU_DEVICE}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
    -v "${repo_root}:/work"
    -v thyllore-docker-target:/work/.docker-target
    -w /work
)

if [ "$#" -eq 0 ]; then
    exec docker run -it "${run_args[@]}" "${image}" bash
fi

exec docker run "${run_args[@]}" "${image}" \
    cargo run --release --bin thyllore-render-usd -- "$@"
