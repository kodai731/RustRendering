"""Measure how closely rendered flame frames match the pillar_ref_seq reference.

Usage:
  python scripts/flame_ref_match.py --render <png|dir> [--crop x0,y0,x1,y1] [--json out.json]
  python scripts/flame_ref_match.py --ref-only

Metrics (silhouette = R>90 && R-B>40, lum = 0.299R+0.587G+0.114B):
  i   lum p10/50/90 within +-20% of reference
  ii  fraction of silhouette pixels with lum<80 within +-8 pt
  iii height-band (10 bands) mean/p90 luminance profile correlation >= 0.8
  iv  G/R and B/R vs luminance-bin curves, RMS difference <= 0.08
  v   spatial contrast spectrum (band-pass std / mean, width-normalised), RMS log2 ratio <= 0.5
  vi  bright coherence: largest bright (>=p70) component share of bright pixels within +-0.15,
      bright fragment count per 1000 bright px ratio within [0.5, 2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REF_DIR = Path(__file__).resolve().parents[1] / "assets/textures/flames/pillar_ref_seq"
LUM_BINS = [(40, 80), (80, 120), (120, 160), (160, 200), (200, 256)]
BAND_COUNT = 10
DARK_LUM = 80.0
NORMALISED_WIDTH = 64.0
CONTRAST_SIGMAS = [1.0, 2.0, 4.0, 8.0, 16.0]
BRIGHT_PERCENTILE = 70.0


def load_rgb(path, crop=None):
    image = Image.open(path).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    return np.asarray(image).astype(np.float64)


def silhouette_mask(rgb):
    r, b = rgb[:, :, 0], rgb[:, :, 2]
    return (r > 90) & (r - b > 40)


def luminance(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def normalise_width(lum, mask):
    widths = mask.sum(axis=1)
    median_width = float(np.median(widths[widths > 0]))
    scale = NORMALISED_WIDTH / max(median_width, 1.0)
    lum_n = ndimage.zoom(lum, scale, order=1)
    mask_n = ndimage.zoom(mask.astype(np.float32), scale, order=1) > 0.5
    return lum_n, mask_n


def contrast_spectrum(lum, mask):
    lum_n, mask_n = normalise_width(lum, mask)
    inner = ndimage.binary_erosion(mask_n, iterations=2)
    if inner.sum() < 50:
        return [np.nan] * len(CONTRAST_SIGMAS)
    filled = np.where(mask_n, lum_n, lum_n[mask_n].mean())
    mean_lum = lum_n[inner].mean()
    spectrum = []
    for sigma in CONTRAST_SIGMAS:
        band = ndimage.gaussian_filter(filled, sigma * 0.5) - ndimage.gaussian_filter(filled, sigma)
        spectrum.append(float(band[inner].std() / mean_lum))
    return spectrum


def bright_coherence(lum, mask):
    lum_n, mask_n = normalise_width(lum, mask)
    if mask_n.sum() < 50:
        return np.nan, np.nan
    bright = mask_n & (lum_n >= np.percentile(lum_n[mask_n], BRIGHT_PERCENTILE))
    labels, count = ndimage.label(bright)
    if count == 0:
        return np.nan, np.nan
    sizes = ndimage.sum(bright, labels, range(1, count + 1))
    largest_share = float(sizes.max() / bright.sum())
    fragments_per_k = float(count / bright.sum() * 1000.0)
    return largest_share, fragments_per_k


def frame_stats(rgb):
    mask = silhouette_mask(rgb)
    if mask.sum() < 100:
        return None
    lum = luminance(rgb)
    largest_share, fragments_per_k = bright_coherence(lum, mask)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max() + 1

    band_mean, band_p90 = [], []
    for band in range(BAND_COUNT):
        ya = y0 + (y1 - y0) * band // BAND_COUNT
        yb = y0 + (y1 - y0) * (band + 1) // BAND_COUNT
        values = lum[ya:yb][mask[ya:yb]]
        band_mean.append(float(values.mean()) if values.size else np.nan)
        band_p90.append(float(np.percentile(values, 90)) if values.size else np.nan)

    gr_curve, br_curve = [], []
    for lo, hi in LUM_BINS:
        selected = mask & (lum >= lo) & (lum < hi)
        if selected.sum() >= 30:
            gr_curve.append(float((g[selected] / r[selected]).mean()))
            br_curve.append(float((b[selected] / r[selected]).mean()))
        else:
            gr_curve.append(np.nan)
            br_curve.append(np.nan)

    return {
        "pixels": int(mask.sum()),
        "lum_p10_50_90": [float(v) for v in np.percentile(lum[mask], [10, 50, 90])],
        "dark_fraction": float((lum[mask] < DARK_LUM).mean()),
        "bright_fraction": float((lum[mask] > 200).mean()),
        "band_mean": band_mean,
        "band_p90": band_p90,
        "gr_by_lum": gr_curve,
        "br_by_lum": br_curve,
        "contrast_spectrum": contrast_spectrum(lum, mask),
        "bright_largest_share": largest_share,
        "bright_fragments_per_k": fragments_per_k,
    }


def aggregate(stat_list):
    keys = ["lum_p10_50_90", "dark_fraction", "bright_fraction", "band_mean", "band_p90", "gr_by_lum", "br_by_lum",
            "contrast_spectrum", "bright_largest_share", "bright_fragments_per_k"]
    out = {"frames": len(stat_list)}
    for key in keys:
        out[key] = np.nanmean(np.array([s[key] for s in stat_list], dtype=float), axis=0).tolist()
    return out


def collect_frames(target):
    target = Path(target)
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() == ".png")
    return [target]


def measure(paths, crop=None):
    stats = []
    for path in paths:
        stat = frame_stats(load_rgb(path, crop))
        if stat is not None:
            stats.append(stat)
    if not stats:
        sys.exit(f"no flame silhouette found in {paths[0]} ...")
    return aggregate(stats)


def nan_corr(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def nan_rms(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def compare(ref, render):
    rows = []
    for name, rv, cv in zip(["lum_p10", "lum_p50", "lum_p90"], ref["lum_p10_50_90"], render["lum_p10_50_90"]):
        ratio = cv / rv
        rows.append((f"i   {name}", f"{rv:.0f}", f"{cv:.0f}", f"ratio {ratio:.2f}", 0.8 <= ratio <= 1.2))
    diff = (render["dark_fraction"] - ref["dark_fraction"]) * 100
    rows.append(("ii  dark(<80) %", f"{ref['dark_fraction'] * 100:.0f}", f"{render['dark_fraction'] * 100:.0f}", f"diff {diff:+.0f}pt", abs(diff) <= 8))
    diff = (render["bright_fraction"] - ref["bright_fraction"]) * 100
    rows.append(("    bright(>200) %", f"{ref['bright_fraction'] * 100:.0f}", f"{render['bright_fraction'] * 100:.0f}", f"diff {diff:+.0f}pt", None))
    for name in ["band_mean", "band_p90"]:
        corr = nan_corr(ref[name], render[name])
        rows.append((f"iii {name} corr", "", "", f"{corr:.2f}", corr >= 0.8))
    for name in ["gr_by_lum", "br_by_lum"]:
        rms = nan_rms(ref[name], render[name])
        rows.append((f"iv  {name} rms", "", "", f"{rms:.3f}", rms <= 0.08))
    log_ratio = np.log2(np.array(render["contrast_spectrum"]) / np.array(ref["contrast_spectrum"]))
    rms = float(np.sqrt(np.nanmean(log_ratio ** 2)))
    rows.append(("v   contrast log2 rms", "", "", f"{rms:.2f}", rms <= 0.5))
    diff = render["bright_largest_share"] - ref["bright_largest_share"]
    rows.append(("vi  bright largest share", f"{ref['bright_largest_share']:.2f}", f"{render['bright_largest_share']:.2f}",
                 f"diff {diff:+.2f}", abs(diff) <= 0.15))
    ratio = render["bright_fragments_per_k"] / ref["bright_fragments_per_k"]
    rows.append(("vi  bright fragments/k", f"{ref['bright_fragments_per_k']:.1f}", f"{render['bright_fragments_per_k']:.1f}",
                 f"ratio {ratio:.2f}", 0.5 <= ratio <= 2.0))
    return rows


def print_profiles(ref, render):
    print("\nheight bands (top -> bottom), lum mean / p90:")
    for band in range(BAND_COUNT):
        rm, rp = ref["band_mean"][band], ref["band_p90"][band]
        cm, cp = render["band_mean"][band], render["band_p90"][band]
        print(f"  band {band}: ref {rm:5.0f}/{rp:5.0f}   render {cm:5.0f}/{cp:5.0f}")
    print("\ncontrast spectrum (band-pass std / mean, sigma in px at column width 64):")
    for sigma, rv, cv in zip(CONTRAST_SIGMAS, ref["contrast_spectrum"], render["contrast_spectrum"]):
        print(f"  sigma {sigma:4.1f} (1/{NORMALISED_WIDTH / sigma:.0f} width): ref {rv:.3f} render {cv:.3f}  x{cv / rv:.2f}")
    print("\nG/R and B/R by lum bin:")
    for (lo, hi), rg, cg, rb, cb in zip(LUM_BINS, ref["gr_by_lum"], render["gr_by_lum"], ref["br_by_lum"], render["br_by_lum"]):
        print(f"  lum {lo:3d}-{hi:3d}: G/R ref {rg:.2f} render {cg:.2f} | B/R ref {rb:.2f} render {cb:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", help="rendered png or directory of pngs")
    parser.add_argument("--crop", help="x0,y0,x1,y1 crop applied to every rendered png")
    parser.add_argument("--ref-dir", default=str(REF_DIR))
    parser.add_argument("--ref-only", action="store_true")
    parser.add_argument("--json", help="write reference/render/gate results to this path")
    args = parser.parse_args()

    ref = measure(collect_frames(args.ref_dir))
    print(f"reference: {ref['frames']} frames, lum p10/50/90 = "
          f"{ref['lum_p10_50_90'][0]:.0f}/{ref['lum_p10_50_90'][1]:.0f}/{ref['lum_p10_50_90'][2]:.0f}, "
          f"dark {ref['dark_fraction'] * 100:.0f}%, bright {ref['bright_fraction'] * 100:.0f}%")
    if args.ref_only or not args.render:
        return

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    render = measure(collect_frames(args.render), crop)
    print(f"render:    {render['frames']} frames, lum p10/50/90 = "
          f"{render['lum_p10_50_90'][0]:.0f}/{render['lum_p10_50_90'][1]:.0f}/{render['lum_p10_50_90'][2]:.0f}, "
          f"dark {render['dark_fraction'] * 100:.0f}%, bright {render['bright_fraction'] * 100:.0f}%")

    rows = compare(ref, render)
    print(f"\n{'gate':<22}{'ref':>6}{'render':>8}  {'value':<14}status")
    passed = 0
    gated = 0
    for name, rv, cv, value, ok in rows:
        status = "-" if ok is None else ("PASS" if ok else "FAIL")
        if ok is not None:
            gated += 1
            passed += int(ok)
        print(f"{name:<22}{rv:>6}{cv:>8}  {value:<14}{status}")
    print(f"\n{passed}/{gated} gates pass")
    print_profiles(ref, render)

    if args.json:
        Path(args.json).write_text(json.dumps({"reference": ref, "render": render,
                                               "gates": [{"name": r[0], "value": r[3], "pass": r[4]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
