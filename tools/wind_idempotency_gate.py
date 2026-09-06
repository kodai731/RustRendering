"""Verify the wind effect renders deterministically and still evolves over time.

Captures the wind probe scene at fixed wind times and checks:
(a) t=0.6 twice -> every compared pixel matches bit for bit.
(b) t=0.6 with --batch-frames 30 and 60 -> every compared pixel matches bit for bit.
(c) t=0.6 vs t=1.6 -> more than 1000 compared pixels differ.

The imgui overlay (x < 300, 600 <= y <= 690 at 2560x1440) jitters between runs, so it is
excluded from every comparison.

    uv run --with numpy --with pillow python3 tools/wind_idempotency_gate.py [--dood]

Exit code 0 = ran to completion (the JSON line on stdout holds pass/fail).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_harness import dood_wrap, engine_env, engine_path, repo_root

HUD_MASK_MAX_X = 300
HUD_MASK_MIN_Y = 600
HUD_MASK_MAX_Y = 690
TIME_EVOLUTION_MIN_DIFF_PIXELS = 1000


def capture(scene: str, camera: str, frames: int, wind_time: float,
            out_path: Path, dood: bool) -> Path:
    """Run the engine once to take a batch screenshot at a fixed wind time."""
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_path),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--batch-wind-time", str(wind_time),
    ]
    if dood:
        command = dood_wrap(command)

    print(f"[wind_idempotency_gate] capture {out_path.name}", file=sys.stderr)
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=600,
        cwd=str(repo_root()), env=engine_env(),
    )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    report = json.loads(last_line) if last_line.startswith("{") else {"ok": False, "error": "no JSON"}
    if not report.get("ok"):
        raise SystemExit(f"capture failed ({out_path.name}): {report.get('error')}\n{proc.stderr[-2000:]}")
    if not out_path.is_file():
        raise SystemExit(f"capture reported ok but wrote no image: {out_path}")
    return out_path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB")).astype(np.int16)


def build_hud_mask(shape: tuple[int, ...]) -> np.ndarray:
    """True on the overlay pixels that must be skipped when comparing two shots."""
    height, width = shape[0], shape[1]
    mask = np.zeros((height, width), dtype=bool)
    mask[HUD_MASK_MIN_Y:min(HUD_MASK_MAX_Y + 1, height), :min(HUD_MASK_MAX_X, width)] = True
    return mask


def count_diff_pixels(first: np.ndarray, second: np.ndarray, hud_mask: np.ndarray) -> int:
    differs = np.any(first != second, axis=-1)
    return int(np.count_nonzero(differs & ~hud_mask))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the wind effect renders deterministically and still evolves over time.")
    parser.add_argument("--scene", default="assets/scenes/wind_probe.scene.ron")
    parser.add_argument("--camera", default="80,25,4,0,0.3,0")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--alt-frames", type=int, default=60,
                        help="second frame count used by the frame independence check")
    parser.add_argument("--out-dir", default="target/tmp_screens/wind_idempotency_gate")
    parser.add_argument("--dood", action="store_true",
                        help="run the engine through the docker harness")
    parser.add_argument("--engine",
                        help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[wind_idempotency_gate] engine {engine_path()}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = repo_root() / scene_path
    if not scene_path.is_file():
        raise SystemExit(f"scene not found: {scene_path}")
    scene = str(scene_path.relative_to(repo_root()))

    run_a = capture(scene, args.camera, args.frames, 0.6,
                    out_dir / f"t0.6_frames{args.frames}_run1.png", args.dood)
    rerun_a = capture(scene, args.camera, args.frames, 0.6,
                      out_dir / f"t0.6_frames{args.frames}_run2.png", args.dood)
    alt_frames = capture(scene, args.camera, args.alt_frames, 0.6,
                         out_dir / f"t0.6_frames{args.alt_frames}.png", args.dood)
    later_time = capture(scene, args.camera, args.frames, 1.6,
                         out_dir / f"t1.6_frames{args.frames}.png", args.dood)

    baseline = load_rgb(run_a)
    hud_mask = build_hud_mask(baseline.shape)

    repeat_diff_pixels = count_diff_pixels(baseline, load_rgb(rerun_a), hud_mask)
    frames_diff_pixels = count_diff_pixels(baseline, load_rgb(alt_frames), hud_mask)
    time_diff_pixels = count_diff_pixels(baseline, load_rgb(later_time), hud_mask)

    deterministic = repeat_diff_pixels == 0
    frame_independent = frames_diff_pixels == 0
    time_evolved = time_diff_pixels > TIME_EVOLUTION_MIN_DIFF_PIXELS

    result = {
        "ok": True,
        "pass": deterministic and frame_independent and time_evolved,
        "deterministic": deterministic,
        "repeat_diff_pixels": repeat_diff_pixels,
        "frame_independent": frame_independent,
        "frames_diff_pixels": frames_diff_pixels,
        "time_evolved": time_evolved,
        "time_diff_pixels": time_diff_pixels,
        "compared_pixels": int(np.count_nonzero(~hud_mask)),
        "hud_masked_pixels": int(np.count_nonzero(hud_mask)),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
