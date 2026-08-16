"""Horizontal line-coherence score for flame screenshots (construction.md §7.4).

score[y] = |mean_x hp| * (|mean_x hp| / mean_x |hp|), hp = residual after a
9-row vertical moving average, flame pixels only, background-structured pixels
excluded. tip = top 45% of the flame's vertical extent, upper = top 22.5%.
Reports score_p99 per region plus the dominant period (band-limited FFT of the
tip row profile) and its power.

Usage:
  uv run --with pillow --with numpy python3 linescore.py \
      --image shot.png --background bg.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HP_WINDOW = 9
FLAME_DIFF_THRESHOLD = 8.0
BG_STRUCTURE_TOLERANCE = 1.0
BG_DILATE_PX = 9
MIN_ROW_PIXELS = 30
FFT_PERIOD_MIN = 4.0
FFT_PERIOD_MAX = 80.0


def load_gray(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


VIEWPORT_LEVEL = 65.0


def detect_viewport(bg: np.ndarray) -> tuple[int, int, int, int]:
    """3D viewport rectangle: longest row/column runs whose count of pixels at
    the viewport background level (gray 65 +/- 1) exceeds a floor — same logic
    as tools/flame_wallness.py detect_viewport_from_background."""
    mask = np.abs(bg - VIEWPORT_LEVEL) <= 1.0

    def longest_run(counts: np.ndarray, floor: int) -> tuple[int, int]:
        best_start, best_end, best_span = 0, len(counts), 0
        run_start = None
        for i, count in enumerate(counts):
            if count > floor and run_start is None:
                run_start = i
            if count <= floor and run_start is not None:
                if i - run_start > best_span:
                    best_start, best_end, best_span = run_start, i, i - run_start
                run_start = None
        if run_start is not None and len(counts) - run_start > best_span:
            best_start, best_end = run_start, len(counts)
        return best_start, best_end

    y0, y1 = longest_run(mask.sum(axis=1).astype(int), 800)
    x0, x1 = longest_run(mask.sum(axis=0).astype(int), 400)
    return x0, y0, x1, y1


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for axis in (0, 1):
        pad = np.zeros_like(out)
        for shift in range(1, radius + 1):
            pad |= np.roll(out, shift, axis=axis) | np.roll(out, -shift, axis=axis)
        out |= pad
    return out


def moving_average_vertical(img: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(img, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), 0, padded
    )


def analyze(image_path: Path, background_path: Path) -> dict:
    flame_full = load_gray(image_path)
    bg_full = load_gray(background_path)
    x0, y0, x1, y1 = detect_viewport(bg_full)
    flame = flame_full[y0:y1, x0:x1]
    bg = bg_full[y0:y1, x0:x1]

    bg_structured = dilate_bool(
        np.abs(bg - VIEWPORT_LEVEL) > BG_STRUCTURE_TOLERANCE, BG_DILATE_PX
    )
    flame_mask = (flame - bg > FLAME_DIFF_THRESHOLD) & ~bg_structured

    hp = flame - moving_average_vertical(flame, HP_WINDOW)
    rows = flame.shape[0]
    score = np.zeros(rows)
    row_mean = np.zeros(rows)
    valid = np.zeros(rows, dtype=bool)
    for y in range(rows):
        mask = flame_mask[y]
        if mask.sum() < MIN_ROW_PIXELS:
            continue
        values = hp[y][mask]
        mean_abs = np.abs(values).mean()
        mean = values.mean()
        if mean_abs < 1e-6:
            continue
        score[y] = abs(mean) * (abs(mean) / mean_abs)
        row_mean[y] = mean
        valid[y] = True

    flame_rows = np.where(flame_mask.sum(axis=1) >= MIN_ROW_PIXELS)[0]
    if len(flame_rows) == 0:
        return {"error": "no flame pixels"}
    top, bottom = flame_rows.min(), flame_rows.max()
    extent = bottom - top + 1

    def region_stats(fraction: float) -> dict:
        lo, hi = top, top + max(int(extent * fraction), 1)
        region = np.arange(lo, hi)
        region = region[valid[region]]
        if len(region) == 0:
            return {"score_p99": 0.0}
        scores = score[region]
        profile = row_mean[lo:hi].copy()
        profile -= profile.mean()
        spectrum = np.abs(np.fft.rfft(profile)) ** 2
        freqs = np.fft.rfftfreq(len(profile))
        with np.errstate(divide="ignore"):
            periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
        band = (periods >= FFT_PERIOD_MIN) & (periods <= FFT_PERIOD_MAX)
        if band.any():
            peak = np.argmax(spectrum * band)
            dominant_period = float(periods[peak])
            dominant_power = float(spectrum[peak] / len(profile))
        else:
            dominant_period, dominant_power = 0.0, 0.0
        return {
            "score_p99": float(np.percentile(scores, 99)),
            "dominant_period_px": dominant_period,
            "dominant_power": dominant_power,
        }

    return {
        "image": str(image_path),
        "viewport": [int(x0), int(y0), int(x1), int(y1)],
        "flame_rows": [int(top), int(bottom)],
        "tip": region_stats(0.45),
        "upper": region_stats(0.225),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--background", required=True)
    args = parser.parse_args()
    for image in args.image:
        print(json.dumps(analyze(Path(image), Path(args.background))))


if __name__ == "__main__":
    main()
