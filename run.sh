#!/usr/bin/env bash
set -euo pipefail

# Single entry point for every launch flavour. The actual run scripts live in
# scripts/ (and src/ml/worker/); this file only dispatches.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_ENV_FILE="$REPO_ROOT/.config/curve_copilot_full.env"

usage() {
    cat <<EOF
Usage: ./run.sh <command> [args...]

Commands:
  engine [private|degrade|full] [cargo args...]
      Launch the engine (cargo run) with a curve copilot mode.
      full sources .config/curve_copilot_full.env (gitignored) for
      THYLLORE_FEEDBACK_TEST_ENDPOINT / THYLLORE_INGEST_TOKEN.
        ./run.sh engine                              # private (default)
        ./run.sh engine degrade
        ./run.sh engine full --features auto-rig
  blender [--mode degrade|full|private] [scene.blend] [args...]
      Build + install the debug addon and launch Blender
      (scripts/run_blender_debug.sh). The mode maps to the addon build mode
      (degrade=A, full=B, private=C; default: degrade). Sources
      .config/curve_copilot_full.env for THYLLORE_FEEDBACK_TEST_ENDPOINT /
      THYLLORE_INGEST_TOKEN (dev builds always bake the test endpoint).
        ./run.sh blender --mode full
  blender --flame [--skip-build] [--release] [--overlap] [--software-gl] [scene.blend]
      Build the flame addon ZIP, install it into a pristine Docker Blender on
      the NVIDIA GPU and open blender/test.blend with a campfire flame already
      added (scripts/blender/flame/launch.sh). Fresh-install checks use blender-verify:
        ./run.sh blender --flame
        ./run.sh blender-verify --zip dist/thyllore_flame-0.0.1-linux_x86_64.zip
  blend [--scene PATH.blend] [--software-gl] [args...]
      Open a pristine Docker Blender on the NVIDIA GPU with a new empty scene,
      no addon installed (blender/docker/run_gui.sh --no-install).
  blender-verify [--mode degrade|full|private] [--zip PATH] [args...]
      Launch a pristine Blender GUI in Docker with NO addon installed
      (blender/docker/run_gui.sh --no-install). The ZIP is mounted at
      /zips for manual verification via Edit > Preferences > Add-ons >
      Install from Disk. Nothing persists between runs.
        ./run.sh blender-verify --mode private --zip dist/release_download/release-zip-linux_x86_64/thyllore_animation_curve_copilot_private-0.0.1-linux_x86_64.zip
  auto [args...]
      Launch the Claude auto-mode container (scripts/run_auto_mode.sh).
  worker-smoke [worker-url] [args...]
      Smoke-test the deployed feedback worker (src/ml/worker/smoke.sh). Sources the
      full-mode env file; WORKER_URL is derived from
      THYLLORE_FEEDBACK_TEST_ENDPOINT when not given.
  help
      Show this help.
EOF
}

command="${1:-help}"
shift || true

case "$command" in
    engine)
        exec bash "$REPO_ROOT/scripts/run_engine.sh" "$@"
        ;;
    blender)
        if [[ "${1:-}" == "--flame" ]]; then
            shift
            exec bash "$REPO_ROOT/scripts/blender/flame/launch.sh" "$@"
        fi
        exec bash "$REPO_ROOT/scripts/run_blender_debug.sh" "$@"
        ;;
    blend)
        exec bash "$REPO_ROOT/blender/docker/run_gui.sh" --no-install "$@"
        ;;
    blender-verify)
        exec bash "$REPO_ROOT/blender/docker/run_gui.sh" --no-install "$@"
        ;;
    auto)
        exec bash "$REPO_ROOT/scripts/run_auto_mode.sh" "$@"
        ;;
    worker-smoke)
        if [[ -r "$FULL_ENV_FILE" ]]; then
            set -a
            # shellcheck disable=SC1090
            source "$FULL_ENV_FILE"
            set +a
        fi
        smoke_endpoint="${THYLLORE_FEEDBACK_TEST_ENDPOINT:-${THYLLORE_FEEDBACK_ENDPOINT:-}}"
        if [[ -z "${WORKER_URL:-}" && -n "$smoke_endpoint" ]]; then
            export WORKER_URL="${smoke_endpoint%/v1/feedback}"
        fi
        exec bash "$REPO_ROOT/src/ml/worker/smoke.sh" "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
