"""Measure the water pass cost at several scissor areas (far / mid / near).

Runs the engine once per camera with --gpu-timings, drops the warmup frames and
reports median / p95 for the water, ray_query and gbuffer passes plus cpu_dt_ms.

    uv run --with numpy python3 tools/water_cost.py [--dood] [--frames 120]
                                                    [--engine target/debug/thyllore-animation]

Exit code 0 = every camera measured (the JSON line on stdout holds the numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_harness import dood_wrap, engine_env, engine_path, repo_root

WARMUP_FRAMES = 10
DEFAULT_CAMERAS = ["far=80,-20,9.0,0,0,0", "mid=80,-20,4.1,0,0,0", "near=80,-20,1.8,0,0,0"]


def parse_cameras(specs: list[str]) -> list[tuple[str, str]]:
    cameras = []
    for spec in specs:
        name, separator, camera = spec.partition("=")
        if not separator or not name or not camera:
            raise SystemExit(f"camera must be name=yaw,pitch,dist,px,py,pz: {spec!r}")
        cameras.append((name, camera))
    return cameras


def capture_timings(name: str, camera: str, scene: str, frames: int, out_dir: Path, dood: bool) -> Path:
    timings_path = out_dir / f"{name}.jsonl"
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_dir / f"{name}.png"),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--gpu-timings", str(timings_path),
    ]
    if dood:
        command = dood_wrap(command)

    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=600,
        cwd=str(repo_root()), env=engine_env(),
    )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    report = json.loads(last_line) if last_line.startswith("{") else {"ok": False, "error": "no JSON"}
    if not report.get("ok"):
        raise SystemExit(f"capture failed ({name}): {report.get('error')}\n{proc.stderr[-2000:]}")

    return timings_path


def load_frames(timings_path: Path) -> list[dict]:
    with timings_path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pass_samples(frames: list[dict], pass_name: str) -> np.ndarray:
    return np.array([float(frame.get("passes", {}).get(pass_name, 0.0)) for frame in frames])


def summarize(name: str, frames: list[dict]) -> dict:
    measured = frames[WARMUP_FRAMES:]
    if not measured:
        raise SystemExit(f"{name}: only {len(frames)} frames captured, need more than {WARMUP_FRAMES}")

    water = pass_samples(measured, "water")
    cpu_dt = np.array([float(frame.get("cpu_dt_ms", 0.0)) for frame in measured])

    return {
        "name": name,
        "water_ms_median": round(float(np.median(water)), 3),
        "water_ms_p95": round(float(np.percentile(water, 95)), 3),
        "ray_query_ms_median": round(float(np.median(pass_samples(measured, "ray_query"))), 3),
        "gbuffer_ms_median": round(float(np.median(pass_samples(measured, "gbuffer"))), 3),
        "cpu_dt_ms_median": round(float(np.median(cpu_dt)), 3),
    }


def write_markdown(path: Path, rows: list[dict], frames: int) -> None:
    lines = [
        f"# water pass cost ({frames} frames, first {WARMUP_FRAMES} dropped)",
        "",
        "| name | water median (ms) | water p95 (ms) | ray_query median (ms) | gbuffer median (ms) | cpu dt median (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['water_ms_median']} | {row['water_ms_p95']} "
            f"| {row['ray_query_ms_median']} | {row['gbuffer_ms_median']} | {row['cpu_dt_ms_median']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the water pass cost per scissor area.")
    parser.add_argument("--scene", default="assets/scenes/default.scene.ron")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out-dir", default="target/tmp_screens/water_cost")
    parser.add_argument("--dood", action="store_true", help="run the engine through the docker harness")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS,
                        help="camera specs as name=yaw,pitch,dist,px,py,pz")
    parser.add_argument("--engine", help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[water_cost] engine {engine_path()}", file=sys.stderr)

    cameras = parse_cameras(args.cameras)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, camera in cameras:
        print(f"[water_cost] capture {name} ({camera})", file=sys.stderr)
        timings_path = capture_timings(name, camera, args.scene, args.frames, out_dir, args.dood)
        rows.append(summarize(name, load_frames(timings_path)))

    write_markdown(out_dir / "water_cost.md", rows, args.frames)
    print(json.dumps({"ok": True, "frames": args.frames, "rows": rows}))


if __name__ == "__main__":
    main()
