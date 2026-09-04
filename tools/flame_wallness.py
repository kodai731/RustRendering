"""Judge whether a screenshot contains a perceivable surface boundary (wall).

Wallness is defined purely as a computer-vision question: does the image
contain a long, smooth, continuous boundary contour that a viewer would read
as a mesh-like surface? Flame-specific quantities are deliberately not used.
Verdict per image: surface_visible.

Capture mode gates the near-wall configs (n1/n2/n3) plus the turbulent
negative-control config (turb) on "no surface visible"; the far config is
reported as reference only. Each config is judged against a same-camera pure
envelope anchor (noise_amplitude=0, boundary_amp=0 = the wall by definition).

Static mode (judge existing images):
    uv run --with pillow --with numpy --with scipy --with opencv-python-headless \
        python3 tools/flame_wallness.py --image a.png b.png [--crop x0,y0,x1,y1]

Capture mode (batch-screenshot configs x {current, envelope anchor}; add
--dood when running from the auto-mode container):
    uv run --with pillow --with numpy --with scipy --with opencv-python-headless \
        python3 tools/flame_wallness.py [--configs n1,n2,n3,turb,far] [--out-dir DIR]

Exit code 0 = analysis ran (verdict is in the JSON line on stdout).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_harness import dood_wrap, engine_env, engine_path, repo_root

import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

BLUR_SIGMA = 4.0
LEVEL_FRACTIONS = (0.1, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85)
MIN_CONTOUR_AREA_FRACTION = 0.01
MIN_RUN_ARC_PX = 100.0
SMOOTH_SIGMA_PX = 32.0
SMOOTH_RESIDUAL_MAX_PX = 3.0
BORDER_MARGIN_PX = 4
GRADIENT_MIN_FRACTION = 0.025
SURFACE_SCORE_FLOOR = 0.35
ENVELOPE_RELATIVE_FACTOR = 0.5

CAMERA_CONFIGS = {
    "n1": ("ring", "180,-8,2.0,0,0.5,-1.5", []),
    "n2": ("ring", "30,-8,0.35", []),
    "n3": ("ring", "180,-10,0.15,0,0.7,-1.4", []),
    "turb": ("ring", "30,-14,0.6,0,0.85,0", []),
    "far": ("ring", "30,-35,4.5", []),
    "cf_near": ("campfire", "82.242,-39.220,1.4315,-0.0906,0.8981,0.0500", ["height=3.06", "radius=1.53"]),
    "cf_far": ("campfire", "82.242,-39.220,1.7485,-0.0616,0.9372,0.0737", ["height=3.06", "radius=1.53"]),
    "ring_wall": ("ring", "-5.134,3.752,1.9324,0.1354,0.4854,-0.0071", []),
    "ring_far": ("ring", "-49.825,11.773,3.8913,0.1186,0.3134,-0.5581", []),
}
GATED_NO_SURFACE = ("n1", "n2", "n3", "turb", "cf_near", "cf_far", "ring_wall")
REFERENCE_ONLY = ("far", "ring_far")
ENVELOPE_OVERRIDES = ["noise_amplitude=0", "boundary_amp=0"]
BACKGROUND_OVERRIDES = ["intensity=0"]
BACKGROUND_MATCH_PX = 8.0


@dataclass
class SurfaceChain:
    score: float
    arc_px: float
    level_fraction: float
    residual_rms_px: float
    mean_gradient: float
    points: np.ndarray


def parse_crop(text: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in text.split(","))
    return x0, y0, x1, y1


def load_gray(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    x0, y0, x1, y1 = crop
    x1 = min(x1, rgb.shape[1])
    y1 = min(y1, rgb.shape[0])
    return rgb[y0:y1, x0:x1].mean(axis=2)


def circular_smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    window = 2 * int(sigma) + 1
    if len(points) <= window:
        return points.copy()
    pad = window
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="wrap")
    smoothed = savgol_filter(padded, window, polyorder=2, axis=0)
    return smoothed[pad:-pad]


def longest_valid_run(valid: np.ndarray) -> tuple[int, int]:
    if valid.all():
        return 0, len(valid)
    doubled = np.concatenate([valid, valid])
    best_start, best_length, run_start = 0, 0, None
    for index, flag in enumerate(doubled):
        if flag and run_start is None:
            run_start = index
        if not flag and run_start is not None:
            length = min(index - run_start, len(valid))
            if length > best_length:
                best_start, best_length = run_start % len(valid), length
            run_start = None
    if run_start is not None:
        length = min(len(doubled) - run_start, len(valid))
        if length > best_length:
            best_start, best_length = run_start % len(valid), length
    return best_start, best_length


def sample_bilinear_clipped(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    rows = np.clip(positions[:, 1], 0, image.shape[0] - 1).astype(np.intp)
    cols = np.clip(positions[:, 0], 0, image.shape[1] - 1).astype(np.intp)
    return image[rows, cols]


def contour_surface_run(
    contour: np.ndarray, gradient: np.ndarray, gradient_min: float, width: int
) -> SurfaceChain | None:
    points = contour.astype(np.float32)
    smoothed = circular_smooth(points, SMOOTH_SIGMA_PX)
    residual = np.linalg.norm(points - smoothed, axis=1)
    height = gradient.shape[0]
    on_border = (
        (points[:, 0] < BORDER_MARGIN_PX)
        | (points[:, 0] > width - 1 - BORDER_MARGIN_PX)
        | (points[:, 1] < BORDER_MARGIN_PX)
        | (points[:, 1] > height - 1 - BORDER_MARGIN_PX)
    )
    valid = (residual <= SMOOTH_RESIDUAL_MAX_PX) & ~on_border
    start, length = longest_valid_run(valid)
    if length < 2:
        return None
    indices = np.arange(start, start + length) % len(points)
    run = points[indices]
    arc = float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())
    if arc < MIN_RUN_ARC_PX:
        return None
    mean_gradient = float(sample_bilinear_clipped(gradient, run).mean())
    if mean_gradient < gradient_min:
        return None
    rms = float(np.sqrt(np.mean(residual[indices] ** 2)))
    return SurfaceChain(arc / width, arc, 0.0, rms, mean_gradient, run)


def matches_background(run: np.ndarray, background_tree: cKDTree | None) -> bool:
    if background_tree is None:
        return False
    distances, _ = background_tree.query(run)
    return float(np.median(distances)) < BACKGROUND_MATCH_PX


def find_surface_chains(
    gray: np.ndarray, background_tree: cKDTree | None = None
) -> list[SurfaceChain]:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), BLUR_SIGMA)
    background = float(np.percentile(blurred, 10))
    peak = float(np.percentile(blurred, 99.5))
    if peak - background < 8.0:
        return []
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy) / 4.0
    gradient_min = GRADIENT_MIN_FRACTION * (peak - background)
    chains: list[SurfaceChain] = []
    min_area = MIN_CONTOUR_AREA_FRACTION * height * width
    for fraction in LEVEL_FRACTIONS:
        level = background + fraction * (peak - background)
        mask = (blurred >= level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            chain = contour_surface_run(contour[:, 0, :], gradient, gradient_min, width)
            if chain is not None and not matches_background(chain.points, background_tree):
                chain.level_fraction = fraction
                chains.append(chain)
    return sorted(chains, key=lambda c: c.score, reverse=True)


def save_overlay(path: Path, gray: np.ndarray, chains: list[SurfaceChain]) -> None:
    canvas = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255)]
    for chain, color in zip(chains[:3], colors):
        cv2.polylines(
            canvas, [chain.points.astype(np.int32)], isClosed=False, color=color, thickness=2
        )
    cv2.imwrite(str(path), canvas)


def analyze_image(
    path: Path, crop: tuple[int, int, int, int], overlay_dir: Path, threshold: float,
    background_tree: cKDTree | None = None,
) -> dict:
    gray = load_gray(path, crop)
    chains = find_surface_chains(gray, background_tree)
    best = chains[0] if chains else None
    overlay_path = overlay_dir / f"{path.stem}_overlay.png"
    save_overlay(overlay_path, gray, chains)
    return {
        "image": str(path),
        "surface_visible": bool(best and best.score >= threshold),
        "surface_score": round(best.score, 3) if best else 0.0,
        "arc_px": round(best.arc_px, 1) if best else 0,
        "level_fraction": best.level_fraction if best else None,
        "residual_rms_px": round(best.residual_rms_px, 2) if best else None,
        "mean_gradient": round(best.mean_gradient, 2) if best else None,
        "candidate_chains": len(chains),
        "overlay": str(overlay_path),
    }


def capture(output: Path, camera: str, overrides: list[str], frames: int, dood: bool, preset: str = "ring", extra_overrides: list[str] | None = None) -> None:
    command = [
        str(engine_path()),
        "--batch-screenshot", str(output),
        "--batch-frames", str(frames),
        "--batch-flame-preset", preset,
        "--batch-flame-mode", "analytic",
        "--batch-camera", camera,
    ]
    if extra_overrides is not None:
        for override in extra_overrides:
            command += ["--batch-flame-set", override]
    for override in overrides:
        command += ["--batch-flame-set", override]
    if dood:
        command = dood_wrap(command)
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=300,
        cwd=str(repo_root()), env=engine_env(),
    )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    result = json.loads(last_line) if last_line.startswith("{") else {"ok": False, "error": "no JSON"}
    if not result.get("ok"):
        raise SystemExit(f"capture failed ({output.name}): {result.get('error')}")


def detect_viewport_from_background(background_png: Path) -> tuple[int, int, int, int] | None:
    """Detect the 3D viewport rectangle from a background image (intensity=0).

    Convert to grayscale, create a mask where pixel value difference from 65 is <= 1,
    find the largest continuous row/column range where the mask has >800 pixels in rows
    and >400 pixels in columns.
    """
    rgb = np.asarray(Image.open(background_png).convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    mask = np.abs(gray - 65.0) <= 1.0

    row_counts = mask.sum(axis=1).astype(int)
    col_counts = mask.sum(axis=0).astype(int)

    best_row_start, best_row_end = 0, len(row_counts)
    best_row_span = 0
    run_start = None
    for i, count in enumerate(row_counts):
        if count > 800 and run_start is None:
            run_start = i
        if count <= 800 and run_start is not None:
            span = i - run_start
            if span > best_row_span:
                best_row_span = span
                best_row_start = run_start
                best_row_end = i
            run_start = None
    if run_start is not None:
        span = len(row_counts) - run_start
        if span > best_row_span:
            best_row_span = span
            best_row_start = run_start
            best_row_end = len(row_counts)

    best_col_start, best_col_end = 0, len(col_counts)
    best_col_span = 0
    run_start = None
    for i, count in enumerate(col_counts):
        if count > 400 and run_start is None:
            run_start = i
        if count <= 400 and run_start is not None:
            span = i - run_start
            if span > best_col_span:
                best_col_span = span
                best_col_start = run_start
                best_col_end = i
            run_start = None
    if run_start is not None:
        span = len(col_counts) - run_start
        if span > best_col_span:
            best_col_span = span
            best_col_start = run_start
            best_col_end = len(col_counts)

    if best_row_span > 0 and best_col_span > 0:
        return (best_col_start, best_row_start, best_col_end, best_row_end)
    return None


def run_capture_mode(config_ids: list[str], out_dir: Path, crop, frames: int, dood: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture a background image (intensity=0) from the first config to detect viewport
    first_preset, first_camera, first_extra = CAMERA_CONFIGS[config_ids[0]]
    background_png = out_dir / "viewport_background.png"
    capture(background_png, first_camera, BACKGROUND_OVERRIDES, frames, dood, first_preset, first_extra)

    # Detect viewport from background image; fallback to crop arg if detection fails
    detected_crop = detect_viewport_from_background(background_png)
    if detected_crop is None and crop is not None:
        detected_crop = crop
    if detected_crop is None:
        raise SystemExit("viewport detection failed and no crop fallback provided")

    configs: dict[str, dict] = {}
    overall_pass = True
    for config_id in config_ids:
        preset, camera, extra_overrides = CAMERA_CONFIGS[config_id]
        current_png = out_dir / f"{config_id}_current.png"
        envelope_png = out_dir / f"{config_id}_envelope.png"
        capture(current_png, camera, [], frames, dood, preset, extra_overrides)
        capture(envelope_png, camera, ENVELOPE_OVERRIDES, frames, dood, preset, extra_overrides)

        background_chains = find_surface_chains(load_gray(background_png, detected_crop))
        background_tree = (
            cKDTree(np.concatenate([chain.points for chain in background_chains]))
            if background_chains else None
        )
        envelope = analyze_image(
            envelope_png, detected_crop, out_dir, SURFACE_SCORE_FLOOR, background_tree
        )
        threshold = max(
            ENVELOPE_RELATIVE_FACTOR * envelope["surface_score"], SURFACE_SCORE_FLOOR
        )
        current = analyze_image(current_png, detected_crop, out_dir, threshold, background_tree)
        current["envelope_score"] = envelope["surface_score"]
        current["threshold"] = round(threshold, 3)
        current["envelope_overlay"] = envelope["overlay"]
        current["gated"] = config_id in GATED_NO_SURFACE
        current["viewport_crop"] = list(detected_crop)
        configs[config_id] = current
        if config_id in GATED_NO_SURFACE and current["surface_visible"]:
            overall_pass = False
    return {"ok": True, "pass": overall_pass, "configs": configs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="*", type=Path)
    parser.add_argument("--crop", type=parse_crop, default=None)
    parser.add_argument("--threshold", type=float, default=SURFACE_SCORE_FLOOR)
    parser.add_argument("--configs", default="n1,n2,n3,turb,far,cf_near,cf_far,ring_wall,ring_far")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dood", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo_root() / "target" / "tmp_screens" / "wallness",
    )
    args = parser.parse_args()

    if args.image:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        results = [
            analyze_image(path, args.crop, args.out_dir, args.threshold)
            for path in args.image
        ]
        print(json.dumps({"ok": True, "images": results}, ensure_ascii=False))
    else:
        report = run_capture_mode(
            args.configs.split(","), args.out_dir, args.crop, args.frames, args.dood
        )
        print(json.dumps(report, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
