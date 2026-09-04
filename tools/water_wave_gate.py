"""Verify the water wave pixel gate (seam continuity and distance aliasing).

Note: Measurements should be performed on a water-only scene without obstructions
(the default scene contains debug primitives).

Captures water debug view 2 (normal visualization) and view 1 (torus mask) for
several cameras, then checks:
- Seam: >10 deg adjacent-normal jumps at the seam <= opposite * 2 + 50 and p99.9 < 10 deg.
- Distance: p99 of adjacent-normal angles at distance 8 <= distance 2 * 2.5.

    uv run --with numpy --with pillow python3 tools/water_wave_gate.py [--dood]

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

CAMERAS = {
    "seam": "260,15,4,0,0,0",
    "opposite": "80,15,4,0,0,0",
    "d2": "80,15,2,0,0,0",
    "d4": "80,15,4,0,0,0",
    "d8": "80,15,8,0,0,0",
}

JUMP_THRESHOLD_DEG = 10.0
MASK_EROSION = 3
MIN_SAMPLE_COUNT = 500


def capture(scene: str, camera: str, frames: int, view: int, out_path: Path, dood: bool,
            water_time: float) -> None:
    """Run the engine once to take a batch screenshot of a water debug view."""
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_path),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--batch-water-time", str(water_time),
        "--batch-water-debug-view", str(view),
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
        raise SystemExit(f"capture failed ({out_path.name}): {report.get('error')}\n{proc.stderr[-2000:]}")
    if not out_path.is_file():
        raise SystemExit(f"capture reported ok but wrote no image: {out_path}")


def capture_all(scene: str, frames: int, out_dir: Path, dood: bool,
                water_time: float) -> dict[str, tuple[Path, Path]]:
    """Capture view 2 and view 1 per camera, reusing shots of an identical camera."""
    shots_by_camera: dict[str, tuple[Path, Path]] = {}
    shots: dict[str, tuple[Path, Path]] = {}

    for name, camera in CAMERAS.items():
        if camera in shots_by_camera:
            shots[name] = shots_by_camera[camera]
            print(f"[water_wave_gate] reuse camera {camera} for {name}", file=sys.stderr)
            continue

        normal_path = out_dir / f"view2_{name}.png"
        mask_path = out_dir / f"view1_{name}.png"
        print(f"[water_wave_gate] capture {normal_path.name} / {mask_path.name}", file=sys.stderr)
        capture(scene, camera, frames, 2, normal_path, dood, water_time)
        capture(scene, camera, frames, 1, mask_path, dood, water_time)

        shots_by_camera[camera] = (normal_path, mask_path)
        shots[name] = (normal_path, mask_path)

    return shots


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB")).astype(np.float64)


def torus_mask(image: np.ndarray) -> np.ndarray:
    """Debug view 1 paints visible torus pixels pure green (2 hits), blue (4 hits) or red (otherwise)."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (
        ((r > 150) & (g < 20) & (b < 20))
        | ((g > 150) & (r < 20) & (b < 20))
        | ((b > 150) & (r < 20) & (g < 20))
    )


def erode_mask(mask: np.ndarray, times: int) -> np.ndarray:
    eroded = mask
    for _ in range(times):
        up = np.pad(eroded, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
        down = np.pad(eroded, ((0, 1), (0, 0)), constant_values=False)[1:, :]
        left = np.pad(eroded, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
        right = np.pad(eroded, ((0, 0), (0, 1)), constant_values=False)[:, 1:]
        eroded = eroded & up & down & left & right
    return eroded


def decode_normals(image: np.ndarray) -> np.ndarray:
    normals = image / 255.0 * 2.0 - 1.0
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(length, 1e-8)


def adjacent_angle_deg(normals: np.ndarray) -> np.ndarray:
    """Angle between each pixel and its right neighbor, shape (H, W - 1)."""
    dot = np.clip(np.sum(normals[:, :-1, :] * normals[:, 1:, :], axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def masked_angles(normal_path: Path, mask_path: Path) -> np.ndarray:
    """Adjacent angles of the pixels whose right neighbor is inside the eroded torus mask too."""
    angles = adjacent_angle_deg(decode_normals(load_rgb(normal_path)))
    mask = erode_mask(torus_mask(load_rgb(mask_path)), MASK_EROSION)
    return angles[mask[:, :-1] & mask[:, 1:]]


def percentile_deg(angles: np.ndarray, percentile: float) -> float:
    return round(float(np.percentile(angles, percentile)), 4)


def evaluate(shots: dict[str, tuple[Path, Path]]) -> dict:
    angles = {name: masked_angles(*paths) for name, paths in shots.items()}
    sample_counts = {name: int(values.size) for name, values in angles.items()}
    too_few = {name: count for name, count in sample_counts.items() if count < MIN_SAMPLE_COUNT}
    if too_few:
        return {
            "ok": False,
            "pass": False,
            "error": f"too few torus samples (min {MIN_SAMPLE_COUNT}): {too_few}",
            "sample_counts": sample_counts,
        }

    seam_angles = angles["seam"]
    opposite_angles = angles["opposite"]

    seam_jump_px = int(np.count_nonzero(seam_angles > JUMP_THRESHOLD_DEG))
    opposite_jump_px = int(np.count_nonzero(opposite_angles > JUMP_THRESHOLD_DEG))
    seam_p999_deg = percentile_deg(seam_angles, 99.9)
    seam_pass = bool(
        seam_jump_px <= opposite_jump_px * 2 + 50 and seam_p999_deg < JUMP_THRESHOLD_DEG
    )

    p99_deg = {name: percentile_deg(angles[name], 99.0) for name in ("d2", "d4", "d8")}
    ratio_d8_d2 = round(p99_deg["d8"] / p99_deg["d2"], 4) if p99_deg["d2"] > 0 else 0.0
    distance_pass = bool(p99_deg["d8"] <= p99_deg["d2"] * 2.5)

    return {
        "ok": True,
        "pass": bool(seam_pass and distance_pass),
        "sample_counts": sample_counts,
        "seam": {
            "seam_jump_px": seam_jump_px,
            "opposite_jump_px": opposite_jump_px,
            "seam_p999_deg": seam_p999_deg,
            "pass": seam_pass,
        },
        "distance": {
            "p99_deg": p99_deg,
            "ratio_d8_d2": ratio_d8_d2,
            "pass": distance_pass,
        },
    }


def write_report(result: dict, out_dir: Path) -> None:
    if not result["ok"]:
        (out_dir / "report.md").write_text(f"# Water Wave Gate\n\nerror: {result['error']}\n")
        return

    seam = result["seam"]
    distance = result["distance"]
    rows = [
        ("seam_jump_px", seam["seam_jump_px"]),
        ("opposite_jump_px", seam["opposite_jump_px"]),
        ("seam_p999_deg", seam["seam_p999_deg"]),
        ("seam_pass", seam["pass"]),
        ("d2_p99_deg", distance["p99_deg"]["d2"]),
        ("d4_p99_deg", distance["p99_deg"]["d4"]),
        ("d8_p99_deg", distance["p99_deg"]["d8"]),
        ("ratio_d8_d2", distance["ratio_d8_d2"]),
        ("distance_pass", distance["pass"]),
        ("pass", result["pass"]),
    ]
    lines = ["# Water Wave Gate", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the water wave pixel gate (seam continuity and distance aliasing).")
    parser.add_argument("--scene", default="assets/scenes/water_probe.scene.ron",
                        help="Scene file to use (measurements should be done on a water-only scene "
                             "without obstructions; default scene contains debug primitives)")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--out-dir", default="target/tmp_screens/water_wave_gate")
    parser.add_argument("--dood", action="store_true",
                        help="run the engine through the docker harness")
    parser.add_argument("--engine",
                        help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    parser.add_argument("--water-time", type=float, default=0.5,
                        help="fixed wave phase passed to --batch-water-time (default 0.5)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[water_wave_gate] engine {engine_path()}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = capture_all(args.scene, args.frames, out_dir, args.dood, args.water_time)
    result = evaluate(shots)

    print(json.dumps(result))
    write_report(result, out_dir)


if __name__ == "__main__":
    main()
