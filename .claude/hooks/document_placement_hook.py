#!/usr/bin/env python3
"""PreToolUse hook: when a .md file is written/edited, surface the document
placement rule so it is auto-read. See .claude/rules/document-placement.md."""
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

path = (data.get("tool_input") or {}).get("file_path", "")
if not path.endswith(".md"):
    sys.exit(0)

reminder = (
    "Writing a .md file. MUST follow .claude/rules/document-placement.md. "
    "Documents go under ${DocumentPath}/Rust_Rendering/ in a TYPE subdirectory: "
    "Design/ for design/spec/decision, ExploreHistory/ for investigation, "
    "IssueHistory/ for bug+fix. Resolve paths from .claude/local/paths.md; never write "
    "directly under Rust_Rendering/. Date-prefix names; a multi-file design uses "
    "Design/<YYYYMMDD>_<topic>/index.md with a same-named child dir. "
    "Read the rule file in full before placing a new document."
)

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": reminder,
    }
}))
sys.exit(0)
