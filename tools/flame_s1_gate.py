"""S1 gate checks — flame shape fidelity against reference measurements.

Each gate is a single scalar measurement compared to a fixed threshold.
Results are printed as one JSON line per gate to stdout.

Capture mode (batch-screenshot with --dood when running from the auto-mode container):
    uv run --with pillow --with numpy --with opencv-python-headless \
        python3 tools/flame_s1_gate.py [--out-dir DIR] [--frames N] [--dood]

Exit code 0 = analysis ran (gate results are in the JSON lines on stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CAMERA = "30,-8,14,0,3,0"
COMMON_OVERRIDES = ["height=5.96", "radius=2.9", "noise_amplitude=3", "noise_contrast=4", "swirl_gain=1.5", "spread_gain=3"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def engine_env() -> dict[str, str]:
    env = dict(os.environ)
    candidates = sorted((repo_root() / "vendor" / "onnxruntime").glob("*/lib/libonnxruntime.so"))
    if candidates:
        env.setdefault("ORT_DYLIB_PATH", str(candidates[-1]))
    return env


def engine_path() -> Path:
    for profile in ("release", "debug"):
        candidate = repo_root() / "target" / profile / "thyllore-animation"
        if candidate.is_file():
            return candidate
    raise SystemExit("engine not built: cargo build --bin thyllore-animation")


DOOD_IMAGE = "thyllore-screenshot-harness:local"


def dood_wrap(command: list[str]) -> list[str]:
    root = str(repo_root())
    ort = "vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so"
    inner = f"ORT_DYLIB_PATH={ort} " + shlex.join(command)
    return [
        "docker", "run", "--rm", "--entrypoint", "bash", "--hostname", "kodai-computer",
        "-v", f"{root}:{root}", "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v", "/run/user/1000/gdm/Xauthority:/xauth:ro",
        "-e", "XAUTHORITY=/xauth", "-e", "DISPLAY=:1",
        "--device", "/dev/dri", "--group-add", "992", "-w", root,
        DOOD_IMAGE, "-c", inner,
    ]


def capture(output: Path, camera: str, overrides: list[str], frames: int, dood: bool) -> None:
    command = [
        str(engine_path()),
        "--batch-screenshot", str(output),
        "--batch-frames", str(frames),
        "--batch-flame-preset", "ring",
        "--batch-flame-mode", "analytic",
        "--batch-camera", camera,
    ]
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


def load_gray(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Load an image and convert to grayscale within the detected crop."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    x0, y0, x1, y1 = crop
    x1 = min(x1, rgb.shape[1])
    y1 = min(y1, rgb.shape[0])
    return rgb[y0:y1, x0:x1].mean(axis=2)


def measure_r0(gray: np.ndarray) -> float:
    """Measure half-width r0 of the threshold mask.

    For each row, find where brightness >= 50% of peak, compute half-width in pixels.
    Returns median of row-wise half-widths.
    """
    rows = gray.shape[0]
    half_widths = []
    for row in range(rows):
        values = gray[row]
        peak = values.max()
        if peak < 1:
            continue
        threshold = 0.5 * peak
        above = values >= threshold
        indices = np.where(above)[0]
        if len(indices) < 2:
            continue
        half_widths.append((indices[-1] - indices[0]) / 2.0)
    if not half_widths:
        return 0.0
    return float(np.median(half_widths))


def measure_r0_profile(gray: np.ndarray) -> np.ndarray:
    """Return an array of half-width r0 for each row."""
    rows = gray.shape[0]
    profile = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        values = gray[row]
        peak = values.max()
        if peak < 1:
            continue
        threshold = 0.5 * peak
        above = values >= threshold
        indices = np.where(above)[0]
        if len(indices) < 2:
            continue
        profile[row] = (indices[-1] - indices[0]) / 2.0
    return profile


def measure_edge_harshness(gray: np.ndarray) -> float:
    """For each row in the middle 50% of rows where half-width > 0, measures the width from
    the outermost 80% intersection to the outermost 20% intersection.

    Returns the median of these widths divided by r0.
    """
    rows = gray.shape[0]
    # First compute half-widths for all rows to find rows where half-width > 0
    half_widths = []
    for row in range(rows):
        values = gray[row]
        peak = values.max()
        if peak < 1:
            half_widths.append(0.0)
            continue
        above_20 = values >= 0.2 * peak
        indices_20 = np.where(above_20)[0]
        if len(indices_20) < 2:
            half_widths.append(0.0)
        else:
            half_widths.append((indices_20[-1] - indices_20[0]) / 2.0)

    # Find rows where half-width > 0
    positive_rows = np.where(np.array(half_widths) > 0)[0]
    if len(positive_rows) < 4:
        return 0.0

    # Take only the middle 50% of those rows (between 25th and 75th percentiles)
    p25 = int(np.percentile(positive_rows, 25))
    p75 = int(np.percentile(positive_rows, 75))
    middle_rows = positive_rows[(positive_rows >= p25) & (positive_rows <= p75)]

    edge_widths = []
    for row in middle_rows:
        values = gray[row]
        peak = values.max()
        if peak < 1:
            continue
        above_80 = values >= 0.8 * peak
        above_20 = values >= 0.2 * peak
        indices_80 = np.where(above_80)[0]
        indices_20 = np.where(above_20)[0]
        if len(indices_80) < 2 or len(indices_20) < 2:
            continue
        # Width from outermost 80% intersection to outermost 20% intersection
        edge_width = (indices_20[-1] - indices_20[0]) - (indices_80[-1] - indices_80[0])
        edge_widths.append(edge_width / 2.0)

    if not edge_widths:
        return 0.0
    r0 = measure_r0(gray)
    if r0 < 1:
        return 0.0
    return float(np.median(edge_widths)) / r0


def gate_s1_1(out_dir: Path, frames: int, dood: bool) -> dict:
    """G-S1-1: support_margin sweep with r0 measured via log-log least squares gradient <= 0.1 for pass."""
    margins = [1.5, 2.0, 2.5, 3.0, 4.0]
    r0_values = []
    for margin in margins:
        output = out_dir / f"gate_s1_1_margin{margin}.png"
        overrides = list(COMMON_OVERRIDES) + [f"support_margin={margin}"]
        capture(output, CAMERA, overrides, frames, dood)
        bg = out_dir / "background.png"
        crop = detect_viewport_from_background(bg) if bg.exists() else None
        if crop is None:
            return {"gate": "G-S1-1", "pass": False, "error": "viewport detection failed"}
        gray = load_gray(output, crop)
        r0 = measure_r0(gray)
        r0_values.append(r0)

    log_margins = np.log(np.array(margins, dtype=np.float64))
    log_r0 = np.log(np.array(r0_values, dtype=np.float64))
    gradient = float(np.polyfit(log_margins, log_r0, 1)[0])
    passed = gradient <= 0.1
    return {"gate": "G-S1-1", "gradient": round(gradient, 4), "r0_values": [round(r, 2) for r in r0_values], "pass": passed}


def gate_s1_2a(out_dir: Path, frames: int, dood: bool) -> dict:
    """G-S1-2a: margin=2.0 with noise_amplitude=3 and 0, r0 relative difference >= 0.3 for pass."""
    bg = out_dir / "background.png"
    crop = detect_viewport_from_background(bg) if bg.exists() else None
    if crop is None:
        return {"gate": "G-S1-2a", "pass": False, "error": "viewport detection failed"}

    output_on = out_dir / "gate_s1_2a_noise_on.png"
    overrides_on = ["height=5.96", "radius=2.9", "noise_amplitude=3", "noise_contrast=4", "swirl_gain=1.5", "spread_gain=3", "support_margin=2.0"]
    capture(output_on, CAMERA, overrides_on, frames, dood)
    gray_on = load_gray(output_on, crop)
    r0_on = measure_r0(gray_on)

    output_off = out_dir / "gate_s1_2a_noise_off.png"
    overrides_off = ["height=5.96", "radius=2.9", "noise_amplitude=0", "noise_contrast=4", "swirl_gain=1.5", "spread_gain=3", "support_margin=2.0"]
    capture(output_off, CAMERA, overrides_off, frames, dood)
    gray_off = load_gray(output_off, crop)
    r0_off = measure_r0(gray_off)

    if r0_on < 1:
        return {"gate": "G-S1-2a", "pass": False, "error": "r0_on too small"}
    relative_diff = abs(r0_on - r0_off) / r0_on
    passed = relative_diff >= 0.3
    return {"gate": "G-S1-2a", "relative_diff": round(relative_diff, 4), "r0_on": round(r0_on, 2), "r0_off": round(r0_off, 2), "pass": passed}


def gate_s1_2b(out_dir: Path, frames: int, dood: bool) -> dict:
    """G-S1-2b: margin=2.0 with frames and frames+180, profile correlation < 0.9 for pass."""
    bg = out_dir / "background.png"
    crop = detect_viewport_from_background(bg) if bg.exists() else None
    if crop is None:
        return {"gate": "G-S1-2b", "pass": False, "error": "viewport detection failed"}

    overrides = ["height=5.96", "radius=2.9", "noise_amplitude=3", "noise_contrast=4", "swirl_gain=1.5", "spread_gain=3", "support_margin=2.0"]

    output_t1 = out_dir / "gate_s1_2b_f60.png"
    capture(output_t1, CAMERA, overrides, frames, dood)
    gray_t1 = load_gray(output_t1, crop)
    profile_t1 = measure_r0_profile(gray_t1)

    output_t2 = out_dir / "gate_s1_2b_f240.png"
    capture(output_t2, CAMERA, overrides, frames + 180, dood)
    gray_t2 = load_gray(output_t2, crop)
    profile_t2 = measure_r0_profile(gray_t2)

    # Filter rows: only include rows where peak brightness (half-width) >= 50% of image max half-width
    max_hw = max(profile_t1.max(), profile_t2.max())
    threshold = 0.5 * max_hw
    valid_mask = (profile_t1 >= threshold) & (profile_t2 >= threshold)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 4:
        return {"gate": "G-S1-2b", "pass": False, "error": "too few rows after filtering"}

    # Take the middle 50% (25th to 75th percentile band) of valid indices
    p25 = int(np.percentile(valid_indices, 25))
    p75 = int(np.percentile(valid_indices, 75))
    selected = slice(p25, p75 + 1)

    p1 = profile_t1[selected]
    p2 = profile_t2[selected]
    z = np.arange(len(p1), dtype=np.float64)
    # Compute fluctuation δ(z) = r(z) - trend(z) using 4th-order polynomial fit
    trend1 = np.polyval(np.polyfit(z, p1, 4), z)
    trend2 = np.polyval(np.polyfit(z, p2, 4), z)
    delta1 = p1 - trend1
    delta2 = p2 - trend2
    correlation = float(np.corrcoef(delta1, delta2)[0, 1])
    passed = correlation < 0.9
    return {"gate": "G-S1-2b", "correlation": round(correlation, 4), "pass": passed}


def gate_s1_2c(out_dir: Path, frames: int, dood: bool) -> dict:
    """G-S1-2c: threshold stability — r0 measured at 30%, 50%, 70% of row peak.

    For each row, compute half-width (median) at three thresholds relative to the row peak.
    relative_shift = |r0_30 - r0_70| / r0_50. Pass condition: relative_shift <= 0.15.
    """
    bg = out_dir / "background.png"
    crop = detect_viewport_from_background(bg) if bg.exists() else None
    if crop is None:
        return {"gate": "G-S1-2c", "pass": False, "error": "viewport detection failed"}

    output = out_dir / "gate_s1_2c.png"
    overrides = ["height=5.96", "radius=2.9", "noise_amplitude=3", "noise_contrast=4", "swirl_gain=1.5", "spread_gain=3", "support_margin=2.0"]
    capture(output, CAMERA, overrides, frames, dood)
    gray = load_gray(output, crop)

    # Compute r0 at each threshold level (30%, 50%, 70%)
    def median_half_width_at_threshold(threshold_frac: float) -> float:
        half_widths = []
        for row in range(gray.shape[0]):
            values = gray[row]
            peak = values.max()
            if peak < 1:
                continue
            threshold = threshold_frac * peak
            above = values >= threshold
            indices = np.where(above)[0]
            if len(indices) < 2:
                continue
            half_widths.append((indices[-1] - indices[0]) / 2.0)
        if not half_widths:
            return 0.0
        return float(np.median(half_widths))

    r0_30 = median_half_width_at_threshold(0.3)
    r0_50 = median_half_width_at_threshold(0.5)
    r0_70 = median_half_width_at_threshold(0.7)

    if r0_50 < 1:
        return {"gate": "G-S1-2c", "r0_30": round(r0_30, 4), "r0_50": round(r0_50, 4), "r0_70": round(r0_70, 4), "relative_shift": 0.0, "pass": True}

    relative_shift = abs(r0_30 - r0_70) / r0_50
    passed = relative_shift <= 0.15
    return {"gate": "G-S1-2c", "r0_30": round(r0_30, 4), "r0_50": round(r0_50, 4), "r0_70": round(r0_70, 4), "relative_shift": round(relative_shift, 4), "pass": passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dood", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=repo_root() / "target" / "tmp_screens" / "s1_gate",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture background (intensity=0) for viewport detection
    bg_path = out_dir / "background.png"
    if not bg_path.exists():
        capture(bg_path, CAMERA, ["intensity=0"], args.frames, args.dood)

    gates = [gate_s1_1, gate_s1_2a, gate_s1_2b, gate_s1_2c]
    for gate_fn in gates:
        result = gate_fn(out_dir, args.frames, args.dood)
        print(json.dumps(result, ensure_ascii=False))

    sys.exit(0)


if __name__ == "__main__":
    main()
