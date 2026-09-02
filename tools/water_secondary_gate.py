"""Verify that ray query and screen-space reflections differ only on reflection hit pixels.

Captures three images for one camera:
- debug view 7: Green = in-screen hit, Red = out-of-screen hit, Blue = miss.
- --batch-water-secondary rayquery
- --batch-water-secondary screenspace

Then checks, inside the torus region only (green + red + blue), that the ray query /
screen-space difference stays on the hit masks (green + red) and leaves the miss mask
(blue) untouched. Pixels outside the torus (imgui overlay etc.) are reported but ignored.

    uv run --with numpy --with pillow python3 tools/water_secondary_gate.py [--dood]

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

DIFF_THRESHOLD = 8.0
MISS_DIFF_FRAC_LIMIT = 0.01
DIFF_INSIDE_HITS_MIN = 0.95


def capture(scene: str, camera: str, frames: int, out_path: Path, dood: bool,
            water_time: float, actions: list[str], view_args: list[str]) -> None:
    """Run the engine once to take a batch screenshot of a water secondary variant."""
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_path),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--batch-water-time", str(water_time),
        "--batch-water-history", "0",
    ]
    for action in actions:
        command += ["--batch-debug-action", action]
    command += view_args
    if dood:
        command = dood_wrap(command)

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


def capture_all(scene: str, camera: str, frames: int, out_dir: Path, dood: bool,
                water_time: float, actions: list[str]) -> dict[str, Path]:
    """Capture the hit type debug view plus the ray query and screen-space shots."""
    shots = {
        "debug": (out_dir / "view7_debug.png", ["--batch-water-debug-view", "7"]),
        "rayquery": (out_dir / "rayquery.png", ["--batch-water-secondary", "rayquery"]),
        "screenspace": (out_dir / "screenspace.png", ["--batch-water-secondary", "screenspace"]),
    }

    for name, (out_path, view_args) in shots.items():
        print(f"[water_secondary_gate] capture {out_path.name}", file=sys.stderr)
        capture(scene, camera, frames, out_path, dood, water_time, actions, view_args)

    return {name: out_path for name, (out_path, _) in shots.items()}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB")).astype(np.float64)


def pure_channel_mask(image: np.ndarray, channel: int) -> np.ndarray:
    """Debug view 7 paints one hit type per pure channel: > 150 on it, < 20 on the others."""
    others = [index for index in range(3) if index != channel]
    return (
        (image[:, :, channel] > 150)
        & (image[:, :, others[0]] < 20)
        & (image[:, :, others[1]] < 20)
    )


def max_channel_diff(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.max(np.abs(first - second), axis=-1)


def mask_metrics(diff: np.ndarray, diff_mask: np.ndarray, mask: np.ndarray) -> dict:
    """Pixel count, fraction of pixels that differ, and mean difference inside one mask."""
    px = int(np.count_nonzero(mask))
    if px == 0:
        return {"px": 0, "diff_frac": 0.0, "mean_diff": 0.0}
    return {
        "px": px,
        "diff_frac": round(float(np.count_nonzero(diff_mask & mask)) / px, 6),
        "mean_diff": round(float(np.mean(diff[mask])), 4),
    }


def evaluate(shots: dict[str, Path]) -> dict:
    debug_image = load_rgb(shots["debug"])
    diff = max_channel_diff(load_rgb(shots["rayquery"]), load_rgb(shots["screenspace"]))
    diff_mask = diff > DIFF_THRESHOLD

    onscreen_mask = pure_channel_mask(debug_image, 1)
    offscreen_mask = pure_channel_mask(debug_image, 0)
    miss_mask = pure_channel_mask(debug_image, 2)

    hits_mask = onscreen_mask | offscreen_mask
    torus_mask = hits_mask | miss_mask

    diff_inside_torus_px = int(np.count_nonzero(diff_mask & torus_mask))
    diff_inside_hits_px = int(np.count_nonzero(diff_mask & hits_mask))
    diff_inside_hits_frac = (
        round(diff_inside_hits_px / diff_inside_torus_px, 6) if diff_inside_torus_px > 0 else 1.0
    )

    miss = mask_metrics(diff, diff_mask, miss_mask)

    return {
        "ok": True,
        "pass": bool(
            miss["diff_frac"] < MISS_DIFF_FRAC_LIMIT
            and diff_inside_hits_frac >= DIFF_INSIDE_HITS_MIN
        ),
        "onscreen": mask_metrics(diff, diff_mask, onscreen_mask),
        "offscreen": mask_metrics(diff, diff_mask, offscreen_mask),
        "miss": miss,
        "torus_px": int(np.count_nonzero(torus_mask)),
        "diff_outside_torus_px": int(np.count_nonzero(diff_mask & ~torus_mask)),
        "diff_inside_hits_frac": diff_inside_hits_frac,
    }


def write_report(result: dict, out_dir: Path) -> None:
    rows = [
        (f"{region}_{metric}", result[region][metric])
        for region in ("onscreen", "offscreen", "miss")
        for metric in ("px", "diff_frac", "mean_diff")
    ]
    rows += [
        ("torus_px", result["torus_px"]),
        ("diff_outside_torus_px", result["diff_outside_torus_px"]),
        ("diff_inside_hits_frac", result["diff_inside_hits_frac"]),
        ("pass", result["pass"]),
    ]
    lines = ["# Water Secondary Gate", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that ray query and screen-space reflections differ only on hit pixels.")
    parser.add_argument("--scene", default="assets/scenes/default.scene.ron")
    parser.add_argument("--camera", default="80,25,4,0,0.3,0")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--out-dir", default="target/tmp_screens/water_secondary_gate")
    parser.add_argument("--dood", action="store_true",
                        help="run the engine through the docker harness")
    parser.add_argument("--engine",
                        help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    parser.add_argument("--water-time", type=float, default=0.5,
                        help="fixed wave phase passed to --batch-water-time (default 0.5)")
    parser.add_argument("--actions", default="spawn_cube,spawn_sphere",
                        help="comma separated --batch-debug-action list")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[water_secondary_gate] engine {engine_path()}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = [action.strip() for action in args.actions.split(",") if action.strip()]
    shots = capture_all(args.scene, args.camera, args.frames, out_dir, args.dood,
                        args.water_time, actions)
    result = evaluate(shots)

    print(json.dumps(result))
    write_report(result, out_dir)


if __name__ == "__main__":
    main()
