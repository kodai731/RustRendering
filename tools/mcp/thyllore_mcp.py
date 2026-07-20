"""Thyllore MCP stdio shim — batch-mode CLI dispatch (Phase A).

Holds no engine logic: every tool call spawns the engine binary fresh, so a
rebuilt engine applies on the next call without /mcp reconnect. The command
surface is defined in
SharedData/document/Rust_Rendering/Design/RemoteControl/20260720_remote_control_architecture.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp package not installed: pip install mcp", file=sys.stderr)
    sys.exit(1)

mcp = FastMCP("thyllore", log_level="WARNING")

_BATCH_TIMEOUT_SECONDS = 300


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _engine_path() -> Path | None:
    for profile in ("release", "debug"):
        candidate = _repo_root() / "target" / profile / "thyllore-animation"
        if candidate.is_file():
            return candidate
    return None


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _engine_env() -> dict[str, str]:
    """Direct binary spawn skips cargo's [env] section, so mirror ORT_DYLIB_PATH here."""
    env = dict(os.environ)
    candidates = sorted(
        (_repo_root() / "vendor" / "onnxruntime").glob("*/lib/libonnxruntime.so")
    )
    if candidates:
        env.setdefault("ORT_DYLIB_PATH", str(candidates[-1]))
    return env


def _run_batch(args: list[str]) -> str:
    engine = _engine_path()
    if engine is None:
        return _error("engine not built: cargo build --bin thyllore-animation")
    try:
        proc = subprocess.run(
            [str(engine), *args],
            capture_output=True,
            text=True,
            timeout=_BATCH_TIMEOUT_SECONDS,
            cwd=str(_repo_root()),
            env=_engine_env(),
        )
    except subprocess.TimeoutExpired:
        return _error(f"engine timed out after {_BATCH_TIMEOUT_SECONDS}s")

    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if last_line.startswith("{"):
        return last_line
    return _error(
        f"engine exited with code {proc.returncode} and no JSON result"
        f" (see log/log_*.txt in the repo)"
    )


@mcp.tool()
def screenshot(output: str = "", frames: int = 120) -> str:
    """Launch Thyllore, render `frames` frames, save a viewport PNG, and exit.

    Returns one JSON object: {"ok": true, "path": "<absolute png path>"} or
    {"ok": false, "error": "..."}. Read the PNG at `path` to inspect the
    rendering. `output` must be an absolute .png path; empty picks a default
    under the repo log/ directory. Each call pays a full engine startup
    (several seconds)."""
    out = output or str(_repo_root() / "log" / f"screenshot_batch_{int(time.time())}.png")
    return _run_batch(["--batch-screenshot", out, "--batch-frames", str(frames)])


@mcp.tool()
def status() -> str:
    """Report whether the engine binary is built and which one a call would use."""
    engine = _engine_path()
    if engine is None:
        return _error("engine not built: cargo build --bin thyllore-animation")
    return json.dumps(
        {"ok": True, "engine": str(engine), "built_at": int(engine.stat().st_mtime)},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run()
