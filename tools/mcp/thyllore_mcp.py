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
import tempfile
import statistics
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
def screenshot(
    output: str = "",
    frames: int = 120,
    flame_mode: str = "",
    flame_steps: int = 0,
    camera: str = "",
) -> str:
    """Launch Thyllore, render `frames` frames, save a viewport PNG, and exit.

    Returns one JSON object: {"ok": true, "path": "<absolute png path>"} or
    {"ok": false, "error": "..."}. Read the PNG at `path` to inspect the
    rendering. `output` must be an absolute .png path; empty picks a default
    under the system tmp directory so reboots reclaim the disk space.
    `flame_mode` optionally overrides the flame integrator
    (analytic|raymarch|thickness|noise); `flame_steps` > 0 overrides the
    raymarch step count; `camera` = "yaw_deg,pitch_deg,distance" orbits the
    camera around the origin. Each call pays a full engine startup (seconds)."""
    default_dir = Path(tempfile.gettempdir()) / "thyllore_screenshots"
    default_dir.mkdir(parents=True, exist_ok=True)
    out = output or str(default_dir / f"screenshot_batch_{int(time.time())}.png")
    args = ["--batch-screenshot", out, "--batch-frames", str(frames)]
    if flame_mode:
        args += ["--batch-flame-mode", flame_mode]
    if flame_steps > 0:
        args += ["--batch-flame-steps", str(flame_steps)]
    if camera:
        args += ["--batch-camera", camera]
    return _run_batch(args)


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


@mcp.tool()
def profile(
    frames: int = 240,
    warmup: int = 60,
    flame_mode: str = "",
    camera: str = "",
    keep_dump: bool = False,
) -> str:
    """Launch Thyllore, render `frames` frames, and profile per-pass GPU timings.

    Returns one JSON object: {"ok": true, "frames_measured": <count>,
    "per_pass": {...}, "frame_total_ms": {...}} or {"ok": false, "error": "..."}.
    The engine writes a JSONL file with per-frame GPU timings; lines where frame
    <= warmup are skipped. For each unique pass label, count/mean/p50/p95/max (ms)
    are computed and sorted by mean descending. frame_total_ms reports the same
    statistics over each frame's total ms across all passes. `flame_mode` and
    `camera` work like in screenshot. Set `keep_dump` to True to keep the JSONL
    dump file (path included in result); it is deleted otherwise."""
    default_dir = Path(tempfile.gettempdir()) / "thyllore_profile"
    default_dir.mkdir(parents=True, exist_ok=True)
    png_path = str(default_dir / f"profile_{int(time.time())}.png")
    jsonl_path = str(default_dir / f"profile_{int(time.time())}.jsonl")

    args = [
        "--batch-screenshot",
        png_path,
        "--batch-frames",
        str(frames),
        "--gpu-timings",
        jsonl_path,
    ]
    if flame_mode:
        args += ["--batch-flame-mode", flame_mode]
    if camera:
        args += ["--batch-camera", camera]

    try:
        result = _run_batch(args)
    except Exception as e:
        return _error(str(e))

    data = json.loads(result)
    if not data.get("ok"):
        return result

    try:
        measured: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("frame", 0) > warmup:
                    measured.append(entry)

        if not measured:
            return _error("No frames measured after warmup")

        pass_timings: dict[str, list[float]] = {}
        frame_totals: list[float] = []

        for entry in measured:
            passes = entry.get("passes", {})
            total = 0.0
            for label, ms in passes.items():
                pass_timings.setdefault(label, []).append(ms)
                total += ms
            frame_totals.append(total)

        def _stats(values: list[float]) -> dict:
            sorted_v = sorted(values)
            n = len(sorted_v)
            return {
                "count": n,
                "mean": round(statistics.mean(sorted_v), 3),
                "p50": round(sorted_v[n // 2], 3),
                "p95": round(sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0], 3),
                "max": round(sorted_v[-1], 3),
            }

        per_pass = {
            label: _stats(values) for label, values in pass_timings.items()
        }
        per_pass_sorted = dict(
            sorted(per_pass.items(), key=lambda item: item[1]["mean"], reverse=True)
        )

        out = {
            "ok": True,
            "frames_measured": len(measured),
            "per_pass": per_pass_sorted,
            "frame_total_ms": _stats(frame_totals),
        }
        if keep_dump:
            out["jsonl_path"] = jsonl_path
        return json.dumps(out, ensure_ascii=False)

    except Exception as e:
        return _error(f"failed to parse timings: {e}")
    finally:
        try:
            os.unlink(png_path)
        except OSError:
            pass
        if not keep_dump:
            try:
                os.unlink(jsonl_path)
            except OSError:
                pass


if __name__ == "__main__":
    mcp.run()
