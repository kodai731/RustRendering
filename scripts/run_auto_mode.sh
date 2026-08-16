#!/usr/bin/env bash
# Launch the ThylloreAnimation auto-mode container (Claude Code, --dangerously-skip-permissions).
#
# The repo, AnimationModelTraining, and SharedData (which contains the document repo) are all
# mounted READ-WRITE at their identical host path, so the absolute paths in
# .claude/local/paths.md resolve unchanged inside the container.
#
# Usage:
#   scripts/run_auto_mode.sh                 # start an interactive session
#   scripts/run_auto_mode.sh --rebuild       # rebuild the image first
#   scripts/run_auto_mode.sh -p "build it"   # pass args straight to claude
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PATHS_FILE="$REPO_ROOT/.claude/local/auto-mode.env"
if [ ! -r "$PATHS_FILE" ]; then
    echo "ERROR: $PATHS_FILE not found." >&2
    echo "  It holds the machine-specific absolute roots (gitignored). See the repo's" >&2
    echo "  committed copy of this script header for the expected variables." >&2
    exit 1
fi
set -a
source "$PATHS_FILE"
set +a
: "${THYLLORE_ROOT:?THYLLORE_ROOT must be set in .claude/local/auto-mode.env}"
: "${ANIM_ML_ROOT:?ANIM_ML_ROOT must be set in .claude/local/auto-mode.env}"
: "${SHARED_DATA_ROOT:?SHARED_DATA_ROOT must be set in .claude/local/auto-mode.env}"

IMAGE_NAME="${AUTO_MODE_IMAGE:-thyllore-auto-mode:latest}"
CONTAINER_NAME="${AUTO_MODE_CONTAINER:-thyllore-auto-mode}"

# Share the host's Claude Code binary instead of installing/updating one in the container.
# It is a standalone ELF (~/.local/share/claude/versions/<ver>); resolve the launcher
# symlink so the container always runs the host's current version.
HOST_CLAUDE_BIN="$(readlink -f "$HOME/.local/bin/claude" 2>/dev/null || true)"
if [ -z "$HOST_CLAUDE_BIN" ] || [ ! -x "$HOST_CLAUDE_BIN" ]; then
    echo "ERROR: host Claude Code binary not found at \$HOME/.local/bin/claude." >&2
    echo "  Install it on the host first: curl -fsSL https://claude.ai/install.sh | bash" >&2
    exit 1
fi

CLAUDE_CONFIG_HOST="${CLAUDE_CONFIG_HOST:-$HOME/.config/thyllore-auto-mode}"
CARGO_REGISTRY_HOST="${CARGO_REGISTRY_HOST:-$HOME/.cache/thyllore-auto-mode/cargo-registry}"
CARGO_TARGET_HOST="${CARGO_TARGET_HOST:-$HOME/.cache/thyllore-auto-mode/target}"
CONTAINER_CLAUDE_CRED="$CLAUDE_CONFIG_HOST/.credentials.json"

mkdir -p "$CLAUDE_CONFIG_HOST" "$CARGO_REGISTRY_HOST" "$CARGO_TARGET_HOST"

# Seed the container's Claude config from the host on first run so the OAuth login and
# ~/.claude.json carry over. Credentials refresh whenever the host copy is newer.
host_cred="$HOME/.claude/.credentials.json"
if [ -r "$host_cred" ] && { [ ! -r "$CONTAINER_CLAUDE_CRED" ] || [ "$host_cred" -nt "$CONTAINER_CLAUDE_CRED" ]; }; then
    cp "$host_cred" "$CONTAINER_CLAUDE_CRED"
    chmod 600 "$CONTAINER_CLAUDE_CRED"
    echo ">>> synced host OAuth credentials -> container claude config"
fi
host_json="$HOME/.claude.json"
persist_json="$CLAUDE_CONFIG_HOST/auto-mode.claude.json"
if [ -r "$host_json" ] && [ ! -f "$persist_json" ]; then
    cp "$host_json" "$persist_json"
    chmod 600 "$persist_json"
    echo ">>> seeded container .claude.json from host (first time only)"
fi

# Mount the host's global Claude instructions read-only so the container Claude obeys the
# same personal CLAUDE.md / rules the user relies on everywhere.
CLAUDE_GLOBAL_MOUNTS=()
[ -r "$HOME/.claude/CLAUDE.md" ] && CLAUDE_GLOBAL_MOUNTS+=(-v "$HOME/.claude/CLAUDE.md:/home/dev/.claude/CLAUDE.md:ro")
[ -d "$HOME/.claude/rules" ] && CLAUDE_GLOBAL_MOUNTS+=(-v "$HOME/.claude/rules:/home/dev/.claude/rules:ro")
[ -d "$HOME/.claude/skills" ] && CLAUDE_GLOBAL_MOUNTS+=(-v "$HOME/.claude/skills:/home/dev/.claude/skills:ro")
[ -d "$HOME/.claude/agents" ] && CLAUDE_GLOBAL_MOUNTS+=(-v "$HOME/.claude/agents:/home/dev/.claude/agents:ro")

NEED_BUILD=0
if [ "${1:-}" = "--rebuild" ]; then
    NEED_BUILD=1
    shift
elif ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    NEED_BUILD=1
fi

if [ "$NEED_BUILD" = "1" ]; then
    echo ">>> Building image $IMAGE_NAME ..."
    docker build \
        -f "$REPO_ROOT/docker/Dockerfile.auto-mode" \
        --build-arg "THYLLORE_ROOT=$THYLLORE_ROOT" \
        -t "$IMAGE_NAME" \
        "$REPO_ROOT"
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Anthropic API key: env wins; else read a host key file; else fall back to the OAuth
# credentials already synced into the container config dir.
ANTHROPIC_KEY_ARG=()
key_file="${ANTHROPIC_API_KEY_FILE:-$HOME/.config/anthropic/api_key}"
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -r "$key_file" ]; then
    ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$key_file")"
fi
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ANTHROPIC_KEY_ARG=(-e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY")
elif [ ! -r "$CONTAINER_CLAUDE_CRED" ]; then
    echo "ERROR: no Claude credentials available." >&2
    echo "  Log in once on host (claude.ai OAuth) so $HOME/.claude/.credentials.json exists," >&2
    echo "  or set ANTHROPIC_API_KEY, or write the key to $key_file" >&2
    exit 1
fi

TTY_FLAGS=(--rm)
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS+=(-it)
elif [ -t 0 ]; then
    TTY_FLAGS+=(-i)
fi

# Best-effort docker-out-of-docker so the agent can dispatch AnimationModelTraining
# training containers on the host daemon.
DOCKER_SOCK_ARG=()
if [ -S /var/run/docker.sock ]; then
    DOCKER_SOCK_ARG=(-v "/var/run/docker.sock:/var/run/docker.sock")
    docker_gid="$(getent group docker | cut -d: -f3 || true)"
    [ -n "$docker_gid" ] && DOCKER_SOCK_ARG+=(--group-add "$docker_gid")
fi

# Best-effort GPU passthrough so Vulkan can reach the host renderer (headless offscreen).
GPU_ARG=()
if [ -d /dev/dri ]; then
    GPU_ARG=(--device /dev/dri)
    render_gid="$(getent group render | cut -d: -f3 || true)"
    [ -n "$render_gid" ] && GPU_ARG+=(--group-add "$render_gid" -e "RENDER_GID=$render_gid")
fi

# Large ML model weights live on the nvme1 disk; mount read-write at its host path when present.
LARGE_MODEL_ARG=()
if [ -n "${LARGE_MODEL_ROOT:-}" ] && [ -d "$LARGE_MODEL_ROOT" ]; then
    LARGE_MODEL_ARG=(-v "$LARGE_MODEL_ROOT:$LARGE_MODEL_ROOT")
fi

exec docker run \
    "${TTY_FLAGS[@]}" \
    --name "$CONTAINER_NAME" \
    --hostname "thyllore-auto" \
    --cap-add NET_ADMIN \
    --cap-add NET_RAW \
    --shm-size 8g \
    --add-host "host.docker.internal:host-gateway" \
    -e AUTO_MODE_FIREWALL="${AUTO_MODE_FIREWALL:-1}" \
    "${ANTHROPIC_KEY_ARG[@]}" \
    "${DOCKER_SOCK_ARG[@]}" \
    "${GPU_ARG[@]}" \
    -v "$THYLLORE_ROOT:$THYLLORE_ROOT" \
    -v "$ANIM_ML_ROOT:$ANIM_ML_ROOT" \
    -v "$SHARED_DATA_ROOT:$SHARED_DATA_ROOT" \
    "${LARGE_MODEL_ARG[@]}" \
    -v "$CARGO_REGISTRY_HOST:/home/dev/.cargo/registry" \
    -v "$CARGO_TARGET_HOST:/home/dev/target-cache" \
    -v "$CLAUDE_CONFIG_HOST:/home/dev/.claude" \
    -v "$HOST_CLAUDE_BIN:/home/dev/.local/bin/claude:ro" \
    "${CLAUDE_GLOBAL_MOUNTS[@]}" \
    -w "$THYLLORE_ROOT" \
    "$IMAGE_NAME" \
    claude --dangerously-skip-permissions \
        --add-dir "$ANIM_ML_ROOT" \
        --add-dir "$SHARED_DATA_ROOT" \
        "$@"
