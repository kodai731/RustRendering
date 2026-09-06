"""Measure the wind pass cost at three camera distances (far / mid / near).

Runs the engine once per camera at a fixed wind time with --gpu-timings, drops the
warmup frames and reports median / p95 for the wind pass plus cpu_dt_ms. The same
cameras also capture a flame scene so the G4 "within +-50% of the flame" target is
recorded next to the wind numbers. Nothing is judged, the values are only reported.

    uv run --with numpy python3 tools/wind_cost.py [--dood] [--frames 120]
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
WIND_TIME = 0.8
CAMERA_ORIENTATION = "80,25"
CAMERA_PIVOT = "0,0.3,0"
DEFAULT_DISTANCES = ["far=12", "mid=6", "near=2.5"]


def parse_cameras(specs: list[str]) -> list[tuple[str, str]]:
    cameras = []
    for spec in specs:
        name, separator, distance = spec.partition("=")
        if not separator or not name or not distance:
            raise SystemExit(f"camera must be name=distance: {spec!r}")
        cameras.append((name, f"{CAMERA_ORIENTATION},{distance},{CAMERA_PIVOT}"))
    return cameras


def resolve_scene(scene: str) -> str:
    scene_path = Path(scene)
    if not scene_path.is_absolute():
        scene_path = repo_root() / scene_path
    if not scene_path.is_file():
        raise SystemExit(f"scene not found: {scene_path}")
    return str(scene_path.relative_to(repo_root()))


def capture_timings(name: str, camera: str, scene: str, frames: int, out_dir: Path, dood: bool) -> Path:
    timings_path = out_dir / f"{name}.jsonl"
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_dir / f"{name}.png"),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--batch-wind-time", str(WIND_TIME),
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


def measured_frames(name: str, frames: list[dict]) -> list[dict]:
    measured = frames[WARMUP_FRAMES:]
    if not measured:
        raise SystemExit(f"{name}: only {len(frames)} frames captured, need more than {WARMUP_FRAMES}")
    return measured


def pass_samples(frames: list[dict], pass_name: str) -> np.ndarray:
    return np.array([float(frame.get("passes", {}).get(pass_name, 0.0)) for frame in frames])


def summarize(name: str, distance: str, wind_frames: list[dict], flame_frames: list[dict]) -> dict:
    wind = pass_samples(wind_frames, "wind")
    flame = pass_samples(flame_frames, "flame")
    wind_median = float(np.median(wind))
    flame_median = float(np.median(flame))

    return {
        "name": name,
        "distance": float(distance),
        "wind_ms_median": round(wind_median, 3),
        "wind_ms_p95": round(float(np.percentile(wind, 95)), 3),
        "wind_cpu_dt_ms_median": round(
            float(np.median([float(frame.get("cpu_dt_ms", 0.0)) for frame in wind_frames])), 3),
        "flame_ms_median": round(flame_median, 3),
        "wind_over_flame": round(wind_median / flame_median, 3) if flame_median > 0.0 else None,
    }


def write_markdown(path: Path, rows: list[dict], frames: int) -> None:
    lines = [
        f"# wind pass cost ({frames} frames, first {WARMUP_FRAMES} dropped, wind time {WIND_TIME})",
        "",
        "| name | distance | wind median (ms) | wind p95 (ms) | wind cpu dt median (ms) "
        "| flame median (ms) | wind / flame |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['distance']} | {row['wind_ms_median']} | {row['wind_ms_p95']} "
            f"| {row['wind_cpu_dt_ms_median']} | {row['flame_ms_median']} | {row['wind_over_flame']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the wind pass cost per camera distance.")
    parser.add_argument("--scene", default="assets/scenes/wind_probe.scene.ron")
    parser.add_argument("--flame-scene", default="assets/scenes/b04t025.scene.ron",
                        help="scene captured with the same cameras as the flame cost reference")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out-dir", default="target/tmp_screens/wind_cost")
    parser.add_argument("--dood", action="store_true", help="run the engine through the docker harness")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_DISTANCES,
                        help="camera specs as name=distance")
    parser.add_argument("--engine", help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[wind_cost] engine {engine_path()}", file=sys.stderr)

    wind_scene = resolve_scene(args.scene)
    flame_scene = resolve_scene(args.flame_scene)
    cameras = parse_cameras(args.cameras)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, camera in cameras:
        print(f"[wind_cost] capture {name} ({camera})", file=sys.stderr)
        wind_timings = capture_timings(f"{name}_wind", camera, wind_scene, args.frames, out_dir, args.dood)
        flame_timings = capture_timings(f"{name}_flame", camera, flame_scene, args.frames, out_dir, args.dood)
        distance = camera.split(",")[2]
        rows.append(summarize(
            name, distance,
            measured_frames(f"{name}_wind", load_frames(wind_timings)),
            measured_frames(f"{name}_flame", load_frames(flame_timings)),
        ))

    write_markdown(out_dir / "wind_cost.md", rows, args.frames)
    print(json.dumps({"ok": True, "frames": args.frames, "wind_time": WIND_TIME, "rows": rows}))


if __name__ == "__main__":
    main()
