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


def _read_json_file(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _run_batch_with_dump(args: list[str], keep_png: bool, png_path: str) -> str:
    """Run a batch invocation that also wrote an anim dump; merge dump into result."""
    dump_path = str(
        Path(tempfile.gettempdir()) / f"thyllore_anim_dump_{int(time.time() * 1000)}.json"
    )
    args = [*args, "--batch-anim-dump", dump_path]
    try:
        result = _run_batch(args)
        data = json.loads(result)
        if not data.get("ok"):
            return result
        dump = _read_json_file(dump_path)
        if dump is None:
            return _error("engine succeeded but wrote no readable anim dump")
        out = {"ok": True, "anim": dump}
        if keep_png:
            out["path"] = data.get("path")
        return json.dumps(out, ensure_ascii=False)
    finally:
        try:
            os.unlink(dump_path)
        except OSError:
            pass
        if not keep_png:
            try:
                os.unlink(png_path)
            except OSError:
                pass


def _batch_base_args(frames: int, camera: str, flame_mode: str, screenshot: bool) -> tuple[list[str], str]:
    if screenshot:
        out_dir = Path(tempfile.gettempdir()) / "thyllore_screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = str(out_dir / f"screenshot_batch_{int(time.time() * 1000)}.png")
    else:
        png_path = str(
            Path(tempfile.gettempdir()) / f"thyllore_anim_{int(time.time() * 1000)}.png"
        )
    args = ["--batch-screenshot", png_path, "--batch-frames", str(max(1, frames))]
    if camera:
        args += ["--batch-camera", camera]
    if flame_mode:
        args += ["--batch-flame-mode", flame_mode]
    return args, png_path


@mcp.tool()
def anim_edit(
    edits: str,
    frames: int = 120,
    screenshot: bool = False,
    camera: str = "",
    flame_mode: str = "",
) -> str:
    """Apply flame animation edits through the engine's production event path,
    render `frames` frames (batch time advances 1/60s per frame, so keyframed
    params animate), and return the resulting animation state as JSON.

    `edits` is a semicolon-separated list of specs:
    - `debug_keys=<seed>`: fill all 16 flame param curves with 4 deterministic
      random keys inside safe ranges (same as the UI "Random Keys (Debug)" button)
    - `key=<param>@<time>=<value>`: insert one key (param = snake_case name,
      e.g. height, intensity, temperature_base_k, wind_x)
    - `clear`: remove all flame scalar curves

    Returns {"ok": true, "anim": {flames, clips, timeline}} — `anim.clips[].scalar_curves`
    holds every curve's keyframes, `anim.flames[].params` the sampled values at the
    final rendered frame (time ≈ frames/60). Set `screenshot` to also keep a PNG
    (path in result). `camera` and `flame_mode` work like in the screenshot tool."""
    args, png_path = _batch_base_args(frames, camera, flame_mode, screenshot)
    specs = [s.strip() for s in edits.split(";") if s.strip()]
    if not specs:
        return _error("edits must contain at least one spec")
    for spec in specs:
        args += ["--batch-anim-edit", spec]
    return _run_batch_with_dump(args, screenshot, png_path)


@mcp.tool()
def anim_state(frames: int = 2, camera: str = "") -> str:
    """Read the engine's animation state without editing anything: launch,
    render `frames` frames, and return {"ok": true, "anim": {flames, clips,
    timeline}}. `anim.flames[]` lists each flame entity with its clip schedule
    and current param values; `anim.clips[]` lists every clip with duration,
    bone track count, and scalar curves (flame keyframes). Use a small `frames`
    (default 2) for a fast state peek, or larger to see values mid-animation."""
    args, png_path = _batch_base_args(frames, camera, "", False)
    return _run_batch_with_dump(args, False, png_path)


@mcp.tool()
def debug_action(
    actions: str,
    frames: int = 120,
    screenshot: bool = True,
    camera: str = "",
) -> str:
    """Execute debug-window actions headlessly and (by default) return a
    screenshot so the effect is visible. `actions` is a semicolon-separated
    list; run `list_debug_actions` for the full set. Examples:
    `view_mode=normal` (G-buffer normal view), `view_mode=object_id`,
    `reset_camera`, `camera_to_model`, `add_flame`, `open_flame_curves`.
    View-mode actions write the same DebugViewState the imgui debug panel
    edits; button actions enqueue the same UIEvents as their buttons.
    Returns {"ok": true, "anim": {...}, "path": "<png>"} (path only when
    `screenshot` is true)."""
    args, png_path = _batch_base_args(frames, camera, "", screenshot)
    names = [s.strip() for s in actions.split(";") if s.strip()]
    if not names:
        return _error("actions must contain at least one action name")
    for name in names:
        args += ["--batch-debug-action", name]
    return _run_batch_with_dump(args, screenshot, png_path)


@mcp.tool()
def list_debug_actions() -> str:
    """List the debug actions `debug_action` accepts, straight from the engine
    binary (fast: exits before window creation)."""
    return _run_batch(["--batch-list-debug-actions"])


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
