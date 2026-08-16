"""Synthetic flame textures with known ground truth for texture-reproduction gates.

Draws 2D images in the same vocabulary as the renderer's forward model: per row
height h, an amplitude A(h), half-width W(h) and color C(h), with the horizontal
cross-section equal to the projected biweight profile
    P(dx) = A(h) * (1 - (dx/W)^2)^2 * sqrt(1 - (dx/W)^2)
which is exactly the shape `project_row` produces, so a correct profile fit can
recover the truth up to global normalization. Truth JSON per image:
    envelope[33]  A(h) at h=i/32 (h=0 flame base = image bottom), max-normalized
    radius[33]    W(h)/W(0) at h=i/32
    color[8]      C(h) rgb at h=(i+0.5)/8
    tilt_deg, noise_sigma

Usage: uv run --with pillow --with numpy python3 tools/flame_texture_testgen.py \
           [--out-dir target/tmp_screens/texrepro/testgen]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH = 256
HEIGHT = 384
ROW0 = 24
ROW1 = 359
BASE_HALFWIDTH_PX = 52.0
SEED = 20260802


def biweight_projection(dx: np.ndarray, half_width: float) -> np.ndarray:
    inside = np.clip(1.0 - (dx / max(half_width, 1e-3)) ** 2, 0.0, None)
    return inside * inside * np.sqrt(inside)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """The engine fit decodes PNGs as sRGB (srgb_to_linear); amplitudes here are
    linear-light, so encode with the inverse transfer to survive the round trip."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def gaussian_bump(h: float, center: float, sigma: float) -> float:
    return math.exp(-(((h - center) / sigma) ** 2))


def color_ramp_rainbow(h: float) -> tuple[float, float, float]:
    stops = [
        (0.0, (0.15, 0.35, 1.0)),
        (0.33, (0.6, 0.2, 0.9)),
        (0.66, (1.0, 0.5, 0.1)),
        (1.0, (1.0, 0.9, 0.2)),
    ]
    for (h0, c0), (h1, c1) in zip(stops, stops[1:]):
        if h <= h1:
            t = 0.0 if h1 == h0 else (h - h0) / (h1 - h0)
            t = min(max(t, 0.0), 1.0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return stops[-1][1]


def color_ramp_orange(h: float) -> tuple[float, float, float]:
    base = (1.0, 0.55, 0.15)
    tip = (1.0, 0.25, 0.05)
    return tuple(a + (b - a) * h for a, b in zip(base, tip))


SPECS: dict[str, dict] = {
    "cone": {
        "amplitude": lambda h: 1.0 - 0.7 * h,
        "radius": lambda h: 1.0 - 0.75 * h,
        "color": color_ramp_orange,
        "tilt_deg": 0.0,
        "noise_sigma": 0.0,
    },
    "twin_peak": {
        "amplitude": lambda h: 0.30
        + 0.70 * gaussian_bump(h, 0.22, 0.13)
        + 0.52 * gaussian_bump(h, 0.68, 0.11),
        "radius": lambda h: 1.0 - 0.6 * h,
        "color": color_ramp_orange,
        "tilt_deg": 0.0,
        "noise_sigma": 0.0,
    },
    "hourglass": {
        "amplitude": lambda h: 1.0 - 0.4 * h,
        "radius": lambda h: (1.0 - 0.62 * gaussian_bump(h, 0.5, 0.16)) * (1.0 - 0.35 * h),
        "color": color_ramp_orange,
        "tilt_deg": 0.0,
        "noise_sigma": 0.0,
    },
    "rainbow": {
        "amplitude": lambda h: 1.0 - 0.5 * h,
        "radius": lambda h: 1.0 - 0.6 * h,
        "color": color_ramp_rainbow,
        "tilt_deg": 0.0,
        "noise_sigma": 0.0,
    },
    "tilted": {
        "amplitude": lambda h: 1.0 - 0.7 * h,
        "radius": lambda h: 1.0 - 0.75 * h,
        "color": color_ramp_orange,
        "tilt_deg": 15.0,
        "noise_sigma": 0.0,
    },
    "noisy_cone_s1": {"base": "cone", "noise_sigma": 0.05},
    "noisy_cone_s2": {"base": "cone", "noise_sigma": 0.15},
    "noisy_cone_s3": {"base": "cone", "noise_sigma": 0.30},
}


def resolve_spec(name: str) -> dict:
    spec = SPECS[name]
    if "base" in spec:
        merged = dict(SPECS[spec["base"]])
        merged["noise_sigma"] = spec["noise_sigma"]
        return merged
    return spec


def render_texture(name: str, seed_offset: int) -> tuple[np.ndarray, dict]:
    spec = resolve_spec(name)
    amp_fn, rad_fn, col_fn = spec["amplitude"], spec["radius"], spec["color"]
    tilt_deg = spec["tilt_deg"]
    sigma = spec["noise_sigma"]

    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    rng = np.random.default_rng(SEED + seed_offset)
    row_mid = 0.5 * (ROW0 + ROW1)
    tan_tilt = math.tan(math.radians(tilt_deg))
    xs = np.arange(WIDTH, dtype=np.float64)

    for r in range(ROW0, ROW1 + 1):
        h = 1.0 - (r - ROW0) / (ROW1 - ROW0)
        amp = max(amp_fn(h), 0.0)
        half_width = max(rad_fn(h), 0.02) * BASE_HALFWIDTH_PX
        xc = WIDTH / 2.0 + tan_tilt * (row_mid - r)
        profile = amp * biweight_projection(xs - xc, half_width)
        if sigma > 0.0:
            profile = profile * np.clip(1.0 + rng.normal(0.0, sigma, WIDTH), 0.0, None)
        rgb = np.array(col_fn(h), dtype=np.float64)
        img[r] = profile[:, None] * rgb[None, :]

    img /= max(img.max(), 1e-6)

    envelope = [max(amp_fn(i / 32.0), 0.0) for i in range(33)]
    env_max = max(max(envelope), 1e-6)
    envelope = [v / env_max for v in envelope]
    radius0 = max(rad_fn(0.0), 1e-3)
    radius = [max(rad_fn(i / 32.0), 0.02) / radius0 for i in range(33)]
    color = [list(col_fn((i + 0.5) / 8.0)) for i in range(8)]
    truth = {
        "envelope": envelope,
        "radius": radius,
        "color": color,
        "tilt_deg": tilt_deg,
        "noise_sigma": sigma,
        "image_rows": [ROW0, ROW1],
        "orientation": "row0=top=tip, h=0 base=image bottom",
    }
    return img, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("target/tmp_screens/texrepro/testgen"))
    parser.add_argument("--ids", nargs="*", default=list(SPECS.keys()))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(args.ids):
        img, truth = render_texture(name, idx)
        png = (linear_to_srgb(img) * 255.0).round().astype(np.uint8)
        Image.fromarray(png, "RGB").save(args.out_dir / f"{name}.png")
        (args.out_dir / f"{name}.truth.json").write_text(json.dumps(truth, indent=1))
        print(f"{name}: png + truth written")


if __name__ == "__main__":
    main()
