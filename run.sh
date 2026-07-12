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
      THYLLORE_FEEDBACK_ENDPOINT / THYLLORE_INGEST_TOKEN.
        ./run.sh engine                              # private (default)
        ./run.sh engine degrade
        ./run.sh engine full --features auto-rig
  blender [scene.blend] [args...]
      Build + install the debug addon and launch Blender
      (scripts/run_blender_debug.sh).
  auto [args...]
      Launch the Claude auto-mode container (scripts/run_auto_mode.sh).
  worker-smoke [worker-url] [args...]
      Smoke-test the deployed feedback worker (src/ml/worker/smoke.sh). Sources the
      full-mode env file; WORKER_URL is derived from
      THYLLORE_FEEDBACK_ENDPOINT when not given.
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
        exec bash "$REPO_ROOT/scripts/run_blender_debug.sh" "$@"
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
        if [[ -z "${WORKER_URL:-}" && -n "${THYLLORE_FEEDBACK_ENDPOINT:-}" ]]; then
            export WORKER_URL="${THYLLORE_FEEDBACK_ENDPOINT%/v1/feedback}"
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
