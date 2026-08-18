"""Judge whether a screenshot sequence contains a temporally anchored transparent slit.

The top-seam artifact is defined purely as a computer-vision question: does the
image contain a dark region that is enclosed by brightness above and below in the
same pixel column (the background shows through a slit inside the object), and
does that slit stay at the same pixels across time phases? Flame-specific
quantities are deliberately not used. Natural shedding blobs and gaps between
tongues move with the animation; an integrator artifact is anchored to a fixed
image region, so temporal persistence is the discriminator.

Detector per phase image:
  1. grayscale viewport crop, UI overlay pixels replaced by background level
     (UI mask = pixels that differ from the flat scene background in an
     intensity=0 background capture)
  2. vertical black-hat: closing with a 1x161 column kernel minus the image =
     min(brightness above, brightness below) - darkness of the pixel
  3. gap mask = response >= 0.12 x luminance range
Across K phase captures (frames 60/90/120): persistent mask = pixels gapped in
>= 2 phases. Seam = largest connected persistent component with area fraction
>= 3.5e-4 of the crop (calibrated: artifact 838 px^2, weak-amp 452, truth and
no-displacement 0 at 1994x855).

Static mode (judge existing phase images of one camera):
    uv run --with pillow --with numpy --with scipy --with opencv-python-headless \
        python3 tools/flame_topseam.py --image f60.png f90.png f120.png \
        [--background bg.png | --crop x0,y0,x1,y1]

Capture mode (per config: K analytic phases + K raymarch control phases; add
--dood when running from the auto-mode container):
    uv run --with pillow --with numpy --with scipy --with opencv-python-headless \
        python3 tools/flame_topseam.py [--configs seam_cf,seam_ring] [--out-dir DIR]

Gate: seam_cf analytic must be seam_visible=false after a fix, with its raymarch
control also false (control seam = detector/scene anomaly). seam_ring is
reference-only: its composition shows persistent enclosed gaps even in the
raymarch truth (static veil structure), so it cannot be gated by this principle
and is reported for trend reading only.
Exit code 0 = analysis ran (verdict is in the JSON line on stdout).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flame_wallness import (  # noqa: E402
    detect_viewport_from_background,
    dood_wrap,
    engine_env,
    engine_path,
    parse_crop,
    repo_root,
)

BLUR_SIGMA = 2.0
VERTICAL_CLOSE_PX = 161
GAP_DEPTH_FRACTION = 0.12
MIN_LUMINANCE_RANGE = 8.0
PHASE_FRAMES = (60, 90, 120)
PERSIST_MIN_PHASES = 2
MIN_COMPONENT_AREA_PX = 100
SEAM_AREA_FRACTION = 3.5e-4
UI_MASK_TOLERANCE = 1.0
UI_MASK_DILATE_PX = 9
RAYMARCH_STEPS = 128

CAMERA_CONFIGS = {
    "seam_cf": (
        "campfire",
        "-3.701,-31.199,1.9324,-0.0679,1.3754,0.1973",
        ["height=2.79", "radius=1.74", "intensity=3.606", "noise_amplitude=1.5"],
    ),
    "seam_ring": ("ring", "-11.723,-37.215,3.5210,0.3393,2.3934,-3.2198", []),
}
GATED_NO_SEAM = ("seam_cf",)
REFERENCE_ONLY = ("seam_ring",)
BACKGROUND_OVERRIDES = ["intensity=0"]


def load_gray(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    x0, y0, x1, y1 = crop
    x1 = min(x1, rgb.shape[1])
    y1 = min(y1, rgb.shape[0])
    return rgb[y0:y1, x0:x1].mean(axis=2)


def ui_mask_from_background(background_gray: np.ndarray) -> np.ndarray:
    flat_level = float(np.median(background_gray))
    raw = (np.abs(background_gray - flat_level) > UI_MASK_TOLERANCE).astype(np.uint8)
    kernel = np.ones((UI_MASK_DILATE_PX, UI_MASK_DILATE_PX), np.uint8)
    return cv2.dilate(raw, kernel).astype(bool)


def enclosed_gap_mask(gray: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
    work = gray.copy()
    if ui_mask is not None:
        outside = work[~ui_mask]
        work[ui_mask] = float(np.percentile(outside, 10)) if outside.size else 0.0
    blurred = cv2.GaussianBlur(work, (0, 0), BLUR_SIGMA)
    background = float(np.percentile(blurred, 10))
    peak = float(np.percentile(blurred, 99.5))
    if peak - background < MIN_LUMINANCE_RANGE:
        return np.zeros(gray.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERTICAL_CLOSE_PX))
    response = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel) - blurred
    if ui_mask is not None:
        response[ui_mask] = 0.0
    return response >= GAP_DEPTH_FRACTION * (peak - background)


def persistent_components(
    phase_grays: list[np.ndarray], ui_mask: np.ndarray | None
) -> list[dict]:
    votes = np.zeros(phase_grays[0].shape, dtype=np.int32)
    for gray in phase_grays:
        votes += enclosed_gap_mask(gray, ui_mask).astype(np.int32)
    need = min(PERSIST_MIN_PHASES, len(phase_grays))
    persistent = (votes >= need).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(persistent, 8)
    components = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area >= MIN_COMPONENT_AREA_PX:
            components.append({"area_px": area, "x": x, "y": y, "w": w, "h": h})
    components.sort(key=lambda c: c["area_px"], reverse=True)
    return components


def save_overlay(
    path: Path, gray: np.ndarray, components: list[dict]
) -> None:
    canvas = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for component in components[:5]:
        x, y = component["x"], component["y"]
        cv2.rectangle(
            canvas, (x - 4, y - 4),
            (x + component["w"] + 4, y + component["h"] + 4), (0, 0, 255), 2,
        )
    cv2.imwrite(str(path), canvas)


def analyze_phases(
    paths: list[Path], crop: tuple[int, int, int, int],
    ui_mask: np.ndarray | None, overlay_path: Path,
) -> dict:
    grays = [load_gray(path, crop) for path in paths]
    components = persistent_components(grays, ui_mask)
    crop_area = grays[0].shape[0] * grays[0].shape[1]
    best_area = components[0]["area_px"] if components else 0
    area_fraction = best_area / crop_area
    save_overlay(overlay_path, grays[-1], components)
    return {
        "phases": [str(path) for path in paths],
        "seam_visible": bool(area_fraction >= SEAM_AREA_FRACTION),
        "seam_area_px": best_area,
        "seam_area_fraction": round(area_fraction, 6),
        "threshold_fraction": SEAM_AREA_FRACTION,
        "persistent_components": components[:5],
        "overlay": str(overlay_path),
    }


def capture(
    output: Path, preset: str, camera: str, overrides: list[str],
    frames: int, mode: str, dood: bool,
) -> None:
    command = [
        str(engine_path()),
        "--batch-screenshot", str(output),
        "--batch-frames", str(frames),
        "--batch-flame-preset", preset,
        "--batch-flame-mode", mode,
        "--batch-camera", camera,
    ]
    if mode == "raymarch":
        command += ["--batch-flame-steps", str(RAYMARCH_STEPS)]
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


def run_capture_mode(
    config_ids: list[str], out_dir: Path,
    crop_fallback: tuple[int, int, int, int] | None, dood: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    first_preset, first_camera, first_extra = CAMERA_CONFIGS[config_ids[0]]
    background_png = out_dir / "viewport_background.png"
    capture(
        background_png, first_preset, first_camera,
        first_extra + BACKGROUND_OVERRIDES, PHASE_FRAMES[0], "analytic", dood,
    )
    crop = detect_viewport_from_background(background_png) or crop_fallback
    if crop is None:
        raise SystemExit("viewport detection failed and no crop fallback provided")
    ui_mask = ui_mask_from_background(load_gray(background_png, crop))

    configs: dict[str, dict] = {}
    overall_pass = True
    for config_id in config_ids:
        preset, camera, extra = CAMERA_CONFIGS[config_id]
        entry: dict[str, dict] = {}
        for mode, tag in (("analytic", "current"), ("raymarch", "control")):
            paths = []
            for frames in PHASE_FRAMES:
                png = out_dir / f"{config_id}_{tag}_f{frames}.png"
                capture(png, preset, camera, extra, frames, mode, dood)
                paths.append(png)
            entry[tag] = analyze_phases(
                paths, crop, ui_mask, out_dir / f"{config_id}_{tag}_overlay.png"
            )
        entry["gated"] = config_id in GATED_NO_SEAM
        entry["viewport_crop"] = list(crop)
        configs[config_id] = entry
        if entry["control"]["seam_visible"]:
            entry["control_anomaly"] = True
        if config_id in GATED_NO_SEAM:
            if entry["current"]["seam_visible"] or entry["control"]["seam_visible"]:
                overall_pass = False
    return {"ok": True, "pass": overall_pass, "configs": configs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="*", type=Path)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--crop", type=parse_crop, default=None)
    parser.add_argument("--configs", default="seam_cf,seam_ring")
    parser.add_argument("--dood", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo_root() / "target" / "tmp_screens" / "topseam_gate",
    )
    args = parser.parse_args()

    if args.image:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        crop = args.crop
        ui_mask = None
        if args.background is not None:
            crop = crop or detect_viewport_from_background(args.background)
            if crop is None:
                raise SystemExit("viewport detection failed on --background")
            ui_mask = ui_mask_from_background(load_gray(args.background, crop))
        if crop is None:
            raise SystemExit("static mode needs --background or --crop")
        result = analyze_phases(
            args.image, crop, ui_mask, args.out_dir / "static_overlay.png"
        )
        print(json.dumps({"ok": True, "images": result}, ensure_ascii=False))
    else:
        report = run_capture_mode(
            args.configs.split(","), args.out_dir, args.crop, args.dood
        )
        print(json.dumps(report, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
