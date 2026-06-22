#!/bin/bash
set -e

if [ "${AUTO_MODE_FIREWALL:-1}" = "1" ]; then
    sudo /usr/local/bin/init-firewall.sh || {
        echo "ERROR: firewall init failed. Aborting auto-mode start." >&2
        exit 1
    }
fi

unset GIT_SSH_COMMAND GIT_ASKPASS SSH_AUTH_SOCK
export GIT_TERMINAL_PROMPT=0

# Claude is the host binary bind-mounted at /home/dev/.local/bin/claude (read-only).
export PATH="/home/dev/.local/bin:$PATH"
echo ">>> claude: $(command -v claude) ($(claude --version 2>/dev/null || echo 'NOT FOUND — check the host claude mount'))"

SETTINGS_TEMPLATE="${AUTO_MODE_CLAUDE_SETTINGS_TEMPLATE:-/etc/auto-mode/claude-settings.json}"
SETTINGS_LIVE="/home/dev/.claude/settings.json"
if [ -f "$SETTINGS_TEMPLATE" ]; then
    mkdir -p "$(dirname "$SETTINGS_LIVE")"
    cp "$SETTINGS_TEMPLATE" "$SETTINGS_LIVE"
fi

PERSIST_CLAUDE_JSON="/home/dev/.claude/auto-mode.claude.json"
LIVE_CLAUDE_JSON="/home/dev/.claude.json"

if [ ! -f "$LIVE_CLAUDE_JSON" ] && [ -f "$PERSIST_CLAUDE_JSON" ]; then
    cp "$PERSIST_CLAUDE_JSON" "$LIVE_CLAUDE_JSON"
fi

cleanup_on_exit() {
    if [ -f "$LIVE_CLAUDE_JSON" ]; then
        cp "$LIVE_CLAUDE_JSON" "$PERSIST_CLAUDE_JSON" 2>/dev/null || true
    fi
}
trap cleanup_on_exit EXIT INT TERM

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    export ANTHROPIC_API_KEY
fi

"$@"
