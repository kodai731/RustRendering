"""Verify depth occlusion for the water pass.

Captures four images with the same camera and a fixed wave phase:
    (a) props_only.png      — normal rendering of a water-less scene + spawn_cube + spawn_sphere
    (b) mask_alone.png      — water debug view 1, no occluders
    (c) mask_occluded.png   — water debug view 1 + spawn_cube + spawn_sphere
    (d) final_occluded.png  — normal rendering + spawn_cube + spawn_sphere (visual reference only)

Computes IoU between the occluded water mask and the expected visible mask
(mask_alone minus the cube pixels of props_only), both eroded by --band pixels
so silhouette aliasing is ignored. Reports JSON on stdout and writes diff.png.

    uv run --with numpy --with pillow python3 tools/water_depth_mask.py [--dood]

Exit code 0 = ran to completion (the JSON line on stdout holds pass/fail).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_harness import dood_wrap, engine_env, engine_path, repo_root

NO_WATER_SCENE = "assets/scenes/_water_depth_mask_nowater.scene.ron"


def capture(scene: str, frames: int, camera: str, out_path: Path, dood: bool,
            water_debug_view: int | None = None,
            debug_actions: list[str] | None = None) -> None:
    """Run the engine once to take a batch screenshot."""
    command = [
        str(engine_path()),
        "--batch-screenshot", str(out_path),
        "--batch-scene", scene,
        "--batch-frames", str(frames),
        "--batch-camera", camera,
        "--batch-water-time", "0",
    ]
    if water_debug_view is not None:
        command += ["--batch-water-debug-view", str(water_debug_view)]
    if debug_actions:
        for action in debug_actions:
            command += ["--batch-debug-action", action]
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


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB")).astype(np.int32)


def water_mask(image: np.ndarray) -> np.ndarray:
    """Debug view 1 paints visible torus pixels pure green (2 hits), blue (4 hits) or red (otherwise)."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (
        ((r > 150) & (g < 20) & (b < 20))
        | ((g > 150) & (r < 20) & (b < 20))
        | ((b > 150) & (r < 20) & (g < 20))
    )


def cube_mask(image: np.ndarray) -> np.ndarray:
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (r > 2 * g) & (r > 2 * b) & (r > 60)


def erode_mask(mask: np.ndarray, band: int) -> np.ndarray:
    eroded = mask
    for _ in range(band):
        up = np.pad(eroded, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
        down = np.pad(eroded, ((0, 1), (0, 0)), constant_values=False)[1:, :]
        left = np.pad(eroded, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
        right = np.pad(eroded, ((0, 0), (0, 1)), constant_values=False)[:, 1:]
        eroded = eroded & up & down & left & right
    return eroded


def write_no_water_scene(scene: str) -> str:
    """Copy the scene RON with its water block removed; engine only accepts paths under assets/scenes."""
    source = Path(scene)
    if not source.is_absolute():
        source = repo_root() / source
    without_water = re.sub(
        r"water: Some\(\(.*?\n    \)\),", "water: None,",
        source.read_text(), flags=re.DOTALL,
    )
    (repo_root() / NO_WATER_SCENE).write_text(without_water)
    return NO_WATER_SCENE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify depth occlusion for the water pass.")
    parser.add_argument("--scene", default="assets/scenes/default.scene.ron")
    parser.add_argument("--camera", default="80,-20,4.1,0,0,0",
                        help="yaw,pitch,dist,px,py,pz")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out-dir", default="target/tmp_screens/water_depth_mask")
    parser.add_argument("--dood", action="store_true",
                        help="run the engine through the docker harness")
    parser.add_argument("--engine",
                        help="engine binary to run (defaults to THYLLORE_ENGINE, then release, then debug)")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="IoU threshold for pass (default 0.99)")
    parser.add_argument("--band", type=int, default=2,
                        help="erode both masks by this many pixels before the IoU (default 2)")
    args = parser.parse_args()

    if args.engine:
        os.environ["THYLLORE_ENGINE"] = args.engine
    print(f"[water_depth_mask] engine {engine_path()}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = args.scene
    frames = args.frames
    camera = args.camera
    debug_actions = ["spawn_cube", "spawn_sphere"]

    no_water_scene = write_no_water_scene(scene)
    try:
        print("[water_depth_mask] capture props_only.png", file=sys.stderr)
        capture(no_water_scene, frames, camera, out_dir / "props_only.png", args.dood,
                debug_actions=debug_actions)

        print("[water_depth_mask] capture mask_alone.png", file=sys.stderr)
        capture(scene, frames, camera, out_dir / "mask_alone.png", args.dood,
                water_debug_view=1)

        print("[water_depth_mask] capture mask_occluded.png", file=sys.stderr)
        capture(scene, frames, camera, out_dir / "mask_occluded.png", args.dood,
                water_debug_view=1, debug_actions=debug_actions)

        print("[water_depth_mask] capture final_occluded.png", file=sys.stderr)
        capture(scene, frames, camera, out_dir / "final_occluded.png", args.dood,
                debug_actions=debug_actions)
    finally:
        (repo_root() / NO_WATER_SCENE).unlink(missing_ok=True)

    m_alone = water_mask(load_rgb(out_dir / "mask_alone.png"))
    m_occluded = water_mask(load_rgb(out_dir / "mask_occluded.png"))
    cube_all = cube_mask(load_rgb(out_dir / "props_only.png"))

    expected_visible = m_alone & ~cube_all

    observed_core = erode_mask(m_occluded, args.band)
    expected_core = erode_mask(expected_visible, args.band)

    intersection = np.count_nonzero(observed_core & expected_core)
    union = np.count_nonzero(observed_core | expected_core)
    iou = intersection / union if union > 0 else 0.0

    mask_alone_px = int(np.count_nonzero(m_alone))
    mask_occluded_px = int(np.count_nonzero(m_occluded))
    cube_px = int(np.count_nonzero(cube_all))
    water_over_cube_px = int(np.count_nonzero(m_occluded & cube_all))

    passed = bool(mask_alone_px > 0 and cube_px > 0 and iou >= args.threshold)

    diff = np.zeros((*m_alone.shape, 3), dtype=np.uint8)
    only_expected = expected_core & ~observed_core
    only_observed = observed_core & ~expected_core
    diff[only_expected] = (0, 255, 0)
    diff[only_observed] = (255, 0, 0)
    Image.fromarray(diff).save(out_dir / "diff.png")

    result = {
        "ok": True,
        "pass": passed,
        "iou": round(iou, 4),
        "mask_alone_px": mask_alone_px,
        "mask_occluded_px": mask_occluded_px,
        "cube_px": cube_px,
        "water_over_cube_px": water_over_cube_px,
        "threshold": args.threshold,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
