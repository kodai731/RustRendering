"""Detect horizontal band lines in flame images for E1/E2 experiment analysis.

Computes the row-average luminance profile of a viewport crop, subtracts its
low-frequency trend (moving average), and extracts peak rows from the high-
frequency residual. Reports each peak's row index, height fraction, amplitude,
and prominence. When multiple images are given, also reports whether peak row
positions are consistent across images (temporal phase invariance).

Static mode:
    uv run --with pillow --with numpy --with scipy --with opencv-python-headless \
        python3 tools/flame_bandlines.py --image a.png b.png \
        [--background bg.png | --crop x0,y0,x1,y1]

Exit code 0 = analysis ran (results in the JSON line on stdout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flame_wallness import (  # noqa: E402
    detect_viewport_from_background,
    engine_env,
    engine_path,
    parse_crop,
    repo_root,
)

# Detection parameters
WINDOW_FRACTION = 0.05  # moving average window as fraction of crop height
NUM_PEAKS = 6           # number of top peaks to report

# UI masking parameters (same as flame_topseam.py)
UI_MASK_TOLERANCE = 1.0
UI_MASK_DILATE_PX = 9


def ui_mask_from_background(background_gray: np.ndarray) -> np.ndarray:
    """Build a boolean mask of the static UI from the background image.

    The background (intensity=0) is flat except where the UI sits. Pixels that
    deviate from the median level are UI; they are dilated to cover the full
    extent of each UI element. Returns True for UI pixels to be excluded."""
    flat_level = float(np.median(background_gray))
    raw = (np.abs(background_gray - flat_level) > UI_MASK_TOLERANCE).astype(np.uint8)
    kernel = np.ones((UI_MASK_DILATE_PX, UI_MASK_DILATE_PX), np.uint8)
    return cv2.dilate(raw, kernel).astype(bool)


def load_gray(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Load an image and return its grayscale viewport crop."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    x0, y0, x1, y1 = crop
    x1 = min(x1, rgb.shape[1])
    y1 = min(y1, rgb.shape[0])
    return rgb[y0:y1, x0:x1].mean(axis=2)


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Compute a moving average with a symmetric window (edge-clamped)."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode="same")


def extract_peaks(residual: np.ndarray, num_peaks: int = NUM_PEAKS) -> list[dict]:
    """Extract the top N peak rows from the residual profile.

    Uses scipy.signal.find_peaks if available, otherwise falls back to manual
    detection by finding local maxima.
    """
    try:
        from scipy.signal import find_peaks as _find_peaks
    except ImportError:
        _find_peaks = None

    # Find local maxima
    if _find_peaks is not None:
        peak_indices, properties = _find_peaks(residual, prominence=True)
    else:
        # Manual local maxima detection: a point is a peak if it's greater than
        # both neighbors (edges are excluded)
        peak_indices = []
        for i in range(1, len(residual) - 1):
            if residual[i] > residual[i - 1] and residual[i] > residual[i + 1]:
                peak_indices.append(i)
        peak_indices = np.array(peak_indices, dtype=int)
        # Compute prominence manually: for each peak, the prominence is the
        # minimum height difference between the peak and the highest minimum
        # on either side before reaching a higher peak
        prominences = []
        for pk in peak_indices:
            left_min = residual[pk]
            for j in range(pk - 1, -1, -1):
                if residual[j] > residual[pk]:
                    break
                left_min = min(left_min, residual[j])
            right_min = residual[pk]
            for j in range(pk + 1, len(residual)):
                if residual[j] > residual[pk]:
                    break
                right_min = min(right_min, residual[j])
            prominences.append(residual[pk] - max(left_min, right_min))
        properties = {"prominences": np.array(prominences)}

    if len(peak_indices) == 0:
        return []

    # Sort by amplitude (residual value) descending and take top N
    amplitudes = residual[peak_indices]
    sorted_order = np.argsort(-amplitudes)
    selected = sorted_order[:num_peaks]

    peaks = []
    for idx in selected:
        pk = int(peak_indices[idx])
        peaks.append({
            "row": pk,
            "height_fraction": round(pk / len(residual), 4),
            "amplitude": round(float(amplitudes[idx]), 4),
            "prominence": round(float(properties["prominences"][idx]), 4),
        })

    return peaks


def analyze_image(path: Path, crop: tuple[int, int, int, int],
                  ui_mask: np.ndarray | None = None) -> dict:
    """Analyze a single image for horizontal band lines.

    If ui_mask is given (boolean array, True = UI pixel to exclude), the row-average
    luminance profile is computed from non-masked pixels only. Rows with fewer than
    2% of crop width in valid pixels get NaN profile values and are excluded from peak
    detection."""
    gray = load_gray(path, crop)
    height, width = gray.shape

    # Row-average luminance profile (UI-masked if ui_mask is given)
    if ui_mask is not None:
        # Per-row: mean of non-masked pixels; NaN if too few valid pixels
        valid_counts = np.sum(~ui_mask, axis=1).astype(np.float64)
        min_valid = 0.02 * width
        profile = np.empty(height, dtype=np.float64)
        for y in range(height):
            if valid_counts[y] < min_valid:
                profile[y] = np.nan
            else:
                profile[y] = float(np.mean(gray[y, ~ui_mask[y]]))
    else:
        profile = np.mean(gray, axis=1)

    # Moving average window size (at least 3) — use nanmean for trend
    window_size = max(3, int(round(WINDOW_FRACTION * height)))
    trend = moving_average(np.nan_to_num(profile), window_size)

    # High-frequency residual (NaN rows become 0 in residual)
    residual = np.abs(np.nan_to_num(profile) - trend)

    # RMS of residual (over non-NaN rows only)
    valid_mask = ~np.isnan(profile)
    if np.sum(valid_mask) > 0:
        residual_rms = round(float(np.sqrt(np.mean(residual[valid_mask] ** 2))), 4)
    else:
        residual_rms = 0.0

    # Extract peaks (only from valid rows — NaN rows are already 0 in residual)
    peaks = extract_peaks(residual)

    # Median number of valid pixels across all rows
    if ui_mask is not None:
        valid_px_median = int(np.median(np.sum(~ui_mask, axis=1)))
    else:
        valid_px_median = width

    return {
        "path": str(path),
        "crop": list(crop),
        "shape": [height, width],
        "residual_rms": residual_rms,
        "valid_px_median": valid_px_median,
        "peaks": peaks,
    }


def check_consistency(results: list[dict]) -> dict | None:
    """Check whether peak row positions are consistent across multiple images.

    Returns a consistency report if there are 2+ images, else None.
    """
    if len(results) < 2:
        return None

    # Collect all peak rows from each image
    all_peak_rows = []
    for r in results:
        all_peak_rows.append(set(p["row"] for p in r["peaks"]))

    # Find the most common peak rows across images
    if not any(all_peak_rows):
        return {"consistent": False, "reason": "no peaks detected in any image"}

    # Count how many images each row appears in
    row_counts: dict[int, int] = {}
    for rows in all_peak_rows:
        for row in rows:
            row_counts[row] = row_counts.get(row, 0) + 1

    if not row_counts:
        return {"consistent": False, "reason": "no peaks detected"}

    # Rows that appear in all images
    n_images = len(results)
    consistent_rows = sorted(r for r, c in row_counts.items() if c == n_images)
    # Rows that appear in most images (>= 50%)
    majority_rows = sorted(r for r, c in row_counts.items() if c >= n_images // 2 + 1)

    return {
        "n_images": n_images,
        "consistent_across_all": consistent_rows,
        "majority_agreement": majority_rows,
        "consistent": len(consistent_rows) > 0 or len(majority_rows) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect horizontal band lines in flame images."
    )
    parser.add_argument("--image", nargs="+", type=Path, help="Image paths to analyze")
    parser.add_argument("--background", type=Path, default=None,
                        help="Background image for viewport detection")
    parser.add_argument("--crop", type=parse_crop, default=None,
                        help="Viewport crop as x0,y0,x1,y1")
    args = parser.parse_args()

    if not args.image:
        parser.error("--image is required")

    # Resolve crop from --background or --crop
    crop = args.crop
    ui_mask = None
    if args.background is not None:
        crop = crop or detect_viewport_from_background(args.background)
        if crop is None:
            raise SystemExit("viewport detection failed on --background")
        # Build UI mask from background image
        ui_mask = ui_mask_from_background(load_gray(args.background, crop))
    if crop is None:
        raise SystemExit("static mode needs --background or --crop")

    # Analyze each image
    results = [analyze_image(path, crop, ui_mask) for path in args.image]

    # Build output
    output: dict = {"ok": True, "images": results}
    consistency = check_consistency(results)
    if consistency is not None:
        output["consistency"] = consistency

    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
