"""Texture-reproduction gates (T1-T6) for the Layer-2 profile bake.

Two-stage verification, decoupled so failures localize:
  (1) parameter gates — did the fit read the intent? Compare the engine dump
      (FlameEffect coefficients / baked profiles) against the testgen truth JSON.
  (2) image gates — did the intent reach the rendered pixels? Run the SAME sym
      extractor on the input texture and on the rendered screenshot and compare
      row profiles (correlation), so no flame-specific absolute value is gated.

Conventions (validated against testgen, see flame-texture-reproduction.md 追補 1):
  - image top = flame tip; sym row r <-> h = 1 - r/63
  - F(h) corresponds to truth envelope/radius as F ~= A/R (max-normalized)
  - envelope inversion divides by luminance of the row CHROMATICITY (max ch = 1)
  - widths measured at 0.15 * row peak with fractional crossing

Usage:
  # static parameter gates against a dump json + truth json
  uv run --with pillow --with numpy python3 tools/flame_texture_repro_test.py \
      --check-dump <wall_probe.json> --truth <name.truth.json>
  # image gate between input texture and rendered screenshot (viewport crop)
  uv run --with pillow --with numpy python3 tools/flame_texture_repro_test.py \
      --check-render <input.png> <render.png> [--crop x,y,w,h]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SYM_ROWS = 64
SYM_COLS = 33


def luminance(c: np.ndarray) -> np.ndarray:
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Mirror of the engine's decode (flame_fit::srgb_to_linear): the fit operates
    on linear-light pixels, so this extractor must too."""
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def extract_sym(img: np.ndarray) -> dict | None:
    """Python mirror of preprocess() in flame_texture_fit.rs (luminance sym 64x33)."""
    imgf = srgb_to_linear(img.astype(np.float64) / 255.0) if img.dtype == np.uint8 else img
    lum = luminance(imgf)
    max_lum = lum.max()
    if max_lum < 1e-6:
        return None
    mask = lum >= max_lum * 0.15
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    row_min, row_max = int(rows[0]), int(rows[-1])
    height, width = lum.shape
    centroids, widths = [], []
    for r in range(row_min, row_max + 1):
        xs = np.where(mask[r])[0]
        if len(xs) > 0:
            centroids.append((np.arange(width) * mask[r]).sum() / max(mask[r].sum(), 1))
            widths.append(float(xs[-1] - xs[0]))
        else:
            centroids.append(width * 0.5)
            widths.append(0.0)
    centroids = np.array(centroids)
    hw = max(widths) * 0.75
    n_rows = row_max - row_min + 1
    sym = np.zeros((SYM_ROWS, SYM_COLS))
    for i in range(SYM_ROWS):
        t = i / (SYM_ROWS - 1.0)
        src_r = min(row_min + int(round(t * n_rows)), row_max)
        c = centroids[src_r - row_min]
        for j in range(SYM_COLS):
            dx = (j / (SYM_COLS - 1.0)) * 2.0 * hw
            xp = int(min(max(round(c + dx), 0), width - 1))
            xm = int(min(max(round(c - dx), 0), width - 1))
            sym[i, j] = (lum[src_r, xp] + lum[src_r, xm]) * 0.5
    return {
        "sym": sym,
        "row_min": row_min,
        "row_max": row_max,
        "imgf": imgf,
        "lum": lum,
    }


def fractional_width(row: np.ndarray, level: float) -> float:
    above = np.where(row >= level)[0]
    if len(above) == 0:
        return 0.0
    j = int(above[-1])
    if j >= len(row) - 1:
        return float(j)
    return j + (row[j] - level) / max(row[j] - row[j + 1], 1e-9)


def invert_profiles(img: np.ndarray) -> dict | None:
    """(F, radius, chroma) via the validated v4 inversion; h ascending, 33 samples."""
    res = extract_sym(img)
    if res is None:
        return None
    sym, rm, rM = res["sym"], res["row_min"], res["row_max"]
    imgf, lum = res["imgf"], res["lum"]
    gmax = sym.max()
    n_rows = rM - rm + 1
    env = np.zeros(SYM_COLS)
    rad = np.zeros(SYM_COLS)
    chroma = np.ones((SYM_COLS, 3))
    for i in range(SYM_COLS):
        h = i / (SYM_COLS - 1.0)
        r = int(round((1.0 - h) * (SYM_ROWS - 1)))
        row = sym[r]
        peak = row.max()
        if peak < 0.05 * gmax:
            continue
        rad[i] = fractional_width(row, 0.15 * peak)
        src_r = min(rm + int(round((r / (SYM_ROWS - 1.0)) * n_rows)), rM)
        rl = lum[src_r]
        sel = rl >= 0.5 * max(rl.max(), 1e-6)
        c = imgf[src_r][sel].mean(axis=0) if sel.any() else np.array([1.0, 1.0, 1.0])
        cn = c / max(c.max(), 1e-6)
        chroma[i] = cn
        env[i] = peak / max(float(luminance(cn[None, :])[0]), 1e-3)
    r0 = rad[0] if rad[0] >= 0.05 * rad.max() else rad.max()
    rad_n = np.clip(rad / max(r0, 1e-3), 0.05, 4.0)
    profile_f = env / np.maximum(rad_n, 0.05)
    profile_f = profile_f / max(profile_f.max(), 1e-6)
    return {"F": profile_f, "radius": rad_n, "chroma": chroma, "sym": sym}


def evaluate_chebyshev8(slots: list[list[float]], h: float) -> float:
    """Chebyshev series on [0,1] packed as 2 vec4 slots (mirror of evaluateChebyshev8)."""
    coeffs = [v for slot in slots for v in slot]
    x = 2.0 * min(max(h, 0.0), 1.0) - 1.0
    t_prev, t_cur = 1.0, x
    total = coeffs[0] * t_prev + coeffs[1] * t_cur
    for k in range(2, len(coeffs)):
        t_next = 2.0 * x * t_cur - t_prev
        total += coeffs[k] * t_next
        t_prev, t_cur = t_cur, t_next
    return total


def normalized_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(a.max(), 1e-6)
    b = b / max(b.max(), 1e-6)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def gate_dump_vs_truth(dump_path: Path, truth_path: Path) -> dict:
    dump = json.loads(dump_path.read_text())
    truth = json.loads(truth_path.read_text())
    flame = dump["flames"][0]
    coeff = flame["coefficients"]
    heights = np.array([i / 32.0 for i in range(33)])

    f_cheb = np.array([evaluate_chebyshev8(coeff["height"], h) for h in heights])
    f_cheb = np.clip(f_cheb, 0.0, None)
    t_env = np.array(truth["envelope"])
    t_rad = np.array(truth["radius"])
    t_f = t_env / np.maximum(t_rad, 0.05)

    # Primary gate: the raw baked LUT measures "did the fit read the intent";
    # the Chebyshev-8 rendering series adds the DESIGNED lowpass truncation on
    # top, so it is reported as info only (render fidelity is gated by T6).
    baked_env = flame.get("baked_envelope")
    if baked_env:
        rms_f = normalized_rms(np.array(baked_env), t_f)
    else:
        rms_f = normalized_rms(f_cheb, t_f)
    result = {
        "rms_F": rms_f,
        "pass_F": rms_f < 0.05,
        "rms_F_cheb_info": normalized_rms(f_cheb, t_f),
    }

    baked_radius = flame.get("baked_radius")
    if baked_radius:
        rms_r = normalized_rms(np.array(baked_radius), t_rad)
        result.update({"rms_radius": rms_r, "pass_radius": rms_r < 0.05})
    baked_color = flame.get("baked_color")
    if baked_color:
        hue_dump = [rgb_hue_angle(c) for c in baked_color]
        hue_truth = [rgb_hue_angle(c) for c in truth["color"]]
        order_ok = hue_order_preserved(hue_dump, hue_truth)
        result.update({"hue_order_preserved": order_ok})
    return result


def rgb_hue_angle(c: list[float]) -> float:
    r, g, b = c[:3]
    return math.atan2(math.sqrt(3.0) * (g - b), 2.0 * r - g - b)


def hue_order_preserved(dump_h: list[float], truth_h: list[float]) -> bool:
    def sign(x):
        return 0 if abs(x) < 0.15 else (1 if x > 0 else -1)

    pairs = list(zip(dump_h, truth_h))
    for (d0, t0), (d1, t1) in zip(pairs, pairs[1:]):
        st = sign(t1 - t0)
        if st != 0 and sign(d1 - d0) == -st:
            return False
    return True


def gate_render_vs_input(
    input_path: Path, render_path: Path, crop: tuple | None, background_path: Path | None = None
) -> dict:
    img_in = np.asarray(Image.open(input_path).convert("RGB"))
    img_re = np.asarray(Image.open(render_path).convert("RGB"))
    if crop:
        x, y, w, h = crop
        img_re = img_re[y : y + h, x : x + w]
    if background_path is not None:
        # The engine viewport background is mid-gray, which floods the 15% mask;
        # subtract an intensity=0 capture (same camera) so only emission remains
        # — the same move flame_topseam.py uses.
        img_bg = np.asarray(Image.open(background_path).convert("RGB"))
        if crop:
            x, y, w, h = crop
            img_bg = img_bg[y : y + h, x : x + w]
        lin_re = srgb_to_linear(img_re.astype(np.float64) / 255.0)
        lin_bg = srgb_to_linear(img_bg.astype(np.float64) / 255.0)
        img_re = np.clip(lin_re - lin_bg, 0.0, 1.0)
    p_in = invert_profiles(img_in)
    p_re = invert_profiles(img_re)
    if p_in is None or p_re is None:
        return {"error": "extraction failed", "pass_corr": False}

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = math.sqrt(float((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 1e-9 else 0.0

    c_f = corr(p_in["F"], p_re["F"])
    c_r = corr(p_in["radius"], p_re["radius"])
    return {
        "corr_F": c_f,
        "corr_radius": c_r,
        "pass_corr": c_f >= 0.9 and c_r >= 0.9,
    }


def parse_crop(text: str) -> tuple:
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-dump", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--check-render", nargs=2, type=Path)
    parser.add_argument("--crop", type=parse_crop, default=None)
    parser.add_argument("--background", type=Path, default=None)
    args = parser.parse_args()

    out: dict = {}
    if args.check_dump:
        if not args.truth:
            parser.error("--check-dump requires --truth")
        out["dump"] = gate_dump_vs_truth(args.check_dump, args.truth)
    if args.check_render:
        out["render"] = gate_render_vs_input(
            args.check_render[0], args.check_render[1], args.crop, args.background
        )
    print(json.dumps(out, indent=1))
    ok = all(v for section in out.values() for k, v in section.items() if k.startswith("pass"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
