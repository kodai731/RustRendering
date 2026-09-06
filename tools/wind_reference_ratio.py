"""Measure wind tornado time evolution ratios against a reference frame.

Captures the wind probe scene at fixed wind times from 0.0 to 2.0 in steps of 0.25,
computes a brightness difference mask against the background (t=0 capture) with
threshold 8/255, and measures:
- column_top_y: the top of the column measured up from the bottom of the screen (1 - y / H)
- mid_width: the width at half the column height as a fraction of screen width (w / W)

Each measurement also carries its ratio against the earliest measurable time, which is what
the G5 record compares with the reference footage. The script only reports; it never judges.

The imgui overlay (x < 300, 600 <= y <= 690 at 2560x1440) jitters between runs, so it is
excluded from every measurement.

    uv run --with numpy --with pillow python3 tools/wind_reference_ratio.py [--dood]

Exit code 0 = ran to completion (the JSON on stdout holds the measurements).
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
DIFF_THRESHOLD = 8
BACKGROUND_TIME = 0.0


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

    print(f"[wind_reference_ratio] capture {out_path.name} (t={wind_time})", file=sys.stderr)
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


def capture_times(scene: str, camera: str, frames: int, times: list[float],
                  out_dir: Path, dood: bool) -> dict[float, Path]:
    """Capture every requested wind time once, keyed by time."""
    captured: dict[float, Path] = {}
    for time in times:
        if time in captured:
            continue
        path = out_dir / f"t{time:.2f}_frames{frames}.png"
        capture(scene, camera, frames, time, path, dood)
        captured[time] = path
    return captured


def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale float values in [0, 1]."""
    with Image.open(path) as image:
        return np.array(image.convert("L")).astype(np.float32) / 255.0


def build_hud_mask(shape: tuple[int, ...]) -> np.ndarray:
    """True on the overlay pixels that must be skipped."""
    height, width = shape[0], shape[1]
    mask = np.zeros((height, width), dtype=bool)
    mask[HUD_MASK_MIN_Y:min(HUD_MASK_MAX_Y + 1, height), :min(HUD_MASK_MAX_X, width)] = True
    return mask


def measure_column(frame: np.ndarray, background: np.ndarray, hud_mask: np.ndarray) -> dict:
    """Measure column top y and mid width from a brightness difference mask.

    The difference mask is where |frame - background| > threshold/255.
    - column_top_y: the highest (smallest y) row with any diff pixels, as fraction of height.
      Measured from bottom of screen (y=0 at bottom), so top of column = 1.0 - smallest_y / H.
    - mid_width: at half the column height, the width of the diff region as fraction of screen width.
    """
    diff_mask = np.abs(frame - background) > DIFF_THRESHOLD / 255.0
    diff_mask &= ~hud_mask

    height, width = diff_mask.shape

    row_has_diff = np.any(diff_mask, axis=1)
    diff_rows = np.where(row_has_diff)[0]

    if len(diff_rows) == 0:
        return {"column_top_y": None, "mid_width": None}

    top_row = int(diff_rows[0])
    column_top_y = 1.0 - top_row / height

    bottom_row = int(diff_rows[-1])

    mid_row = (top_row + bottom_row) // 2

    mid_row_diff = diff_mask[mid_row, :]
    if not np.any(mid_row_diff):
        return {"column_top_y": round(column_top_y, 4), "mid_width": None}

    col_indices = np.where(mid_row_diff)[0]
    left = int(col_indices[0])
    right = int(col_indices[-1])
    mid_width = (right - left + 1) / width

    return {
        "column_top_y": round(column_top_y, 4),
        "mid_width": round(mid_width, 4),
    }


def append_ratios(measurements: list[dict]) -> float | None:
    """Add each measurement's ratio against the earliest measurable time and return that time."""
    baseline = next((m for m in measurements if m["column_top_y"] is not None), None)

    for measurement in measurements:
        for key in ("column_top_y", "mid_width"):
            value = measurement[key]
            start = baseline[key] if baseline is not None else None
            usable = value is not None and start is not None and start > 0.0
            measurement[f"{key}_ratio"] = round(value / start, 4) if usable else None

    return baseline["time"] if baseline is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure wind tornado time evolution ratios against a reference frame.")
    parser.add_argument("--scene", default="assets/scenes/wind_probe.scene.ron")
    parser.add_argument("--camera", default="80,25,4,0,0.3,0")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--times", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                        help="wind times to capture (default: 0.0 to 2.0 in steps of 0.25)")
    parser.add_argument("--out-dir", default="target/tmp_screens/wind_reference_ratio")
    parser.add_argument("--dood", action="store_true",
                        help="run the engine through the docker harness")
    parser.add_argument("--engine",
                        help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[wind_reference_ratio] engine {engine_path()}", file=sys.stderr)

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

    captures = capture_times(scene, args.camera, args.frames,
                             [BACKGROUND_TIME, *args.times], out_dir, args.dood)

    background = load_gray(captures[BACKGROUND_TIME])
    hud_mask = build_hud_mask(background.shape)

    measurements = []
    for time in args.times:
        measurement = measure_column(load_gray(captures[time]), background, hud_mask)
        measurement["time"] = time
        measurements.append(measurement)
        print(f"  t={time:.2f}: {measurement}", file=sys.stderr)

    ratio_baseline_time = append_ratios(measurements)

    result = {
        "ok": True,
        "measurements": measurements,
        "ratio_baseline_time": ratio_baseline_time,
        "compared_pixels": int(np.count_nonzero(~hud_mask)),
        "hud_masked_pixels": int(np.count_nonzero(hud_mask)),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
