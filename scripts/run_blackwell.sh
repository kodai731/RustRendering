#!/usr/bin/env bash
# Run the Thyllore offscreen renderer on the NVIDIA Blackwell GPU inside a
# CUDA+Vulkan container. The container sees only the Blackwell (via CDI), so the
# engine's Vulkan device selection picks it automatically — no engine flag needed.
# Loads any supported model format (USD, glTF, FBX) — the loader dispatches by
# file extension.
#
# Modes:
#   scripts/run_blackwell.sh                  launch the interactive Thyllore app
#   scripts/run_blackwell.sh --model <m> ...  offscreen PNG render (thyllore-render)
#   scripts/run_blackwell.sh shell            interactive shell in the container
#
# The GUI app needs the host display. On X11 allow the container first:
#   xhost +local:
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

for var in THYLLORE_HIDE_BONES THYLLORE_HIDE_UNSKINNED THYLLORE_NO_AUTOFRAME THYLLORE_DEBUG_VIEW; do
    if [ -n "${!var:-}" ]; then
        run_args+=(-e "${var}=${!var}")
    fi
done

if [ "${1:-}" = "shell" ]; then
    exec docker run -it "${run_args[@]}" "${image}" bash
fi

# Offscreen PNG render path (used by compare_blender_thyllore.sh and manual renders).
if [ "$#" -gt 0 ]; then
    exec docker run "${run_args[@]}" "${image}" \
        cargo run --bin thyllore-render -- "$@"
fi

# No arguments: launch the interactive Thyllore app on the host display.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "No DISPLAY or WAYLAND_DISPLAY set — cannot open a window." >&2
    echo "Run from a desktop session, or use 'scripts/run_blackwell.sh --model ...' for offscreen." >&2
    exit 1
fi

# --net=host lets the container reach the host X server / Wayland socket directly.
display_args=(--net=host)

if [ -n "${DISPLAY:-}" ]; then
    display_args+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)
    xauth_host="${XAUTHORITY:-${HOME}/.Xauthority}"
    if [ -f "${xauth_host}" ]; then
        display_args+=(-e "XAUTHORITY=/tmp/.docker.xauth" -v "${xauth_host}:/tmp/.docker.xauth:ro")
    fi
fi

if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    wayland_socket="${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}"
    if [ -S "${wayland_socket}" ]; then
        display_args+=(
            -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}"
            -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
            -v "${wayland_socket}:${wayland_socket}"
        )
    fi
fi

echo "[run_blackwell] launching app. If it fails with a display error, run: xhost +local:" >&2

exec docker run -it "${run_args[@]}" "${display_args[@]}" "${image}" \
    cargo run --bin thyllore-animation
