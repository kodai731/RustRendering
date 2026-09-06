"""Measure how closely rendered flame frames match the pillar_ref_seq_stable reference.

Usage:
  python scripts/flame_ref_match.py --render <png|dir> [--crop x0,y0,x1,y1] [--json out.json]
  python scripts/flame_ref_match.py --ref-only
  python scripts/flame_ref_match.py --render <dir> --temporal [--fps 10]   (sequence gates xiii-xvi)

The reference is measured at its native resolution. Every render frame is first resampled
with an area filter so that its median flame column width equals the reference's median
column width, so the render may be captured at any higher resolution and the comparison
happens at the reference pixel density.

Metrics (silhouette = R>90 && R-B>40, lum = 0.299R+0.587G+0.114B):
  i   lum p10/50/90 within +-20% of reference
  ii  fraction of silhouette pixels with lum<80 within +-8 pt
  iii height-band (10 bands) mean/p90 luminance profile correlation >= 0.8
  iv  G/R and B/R vs luminance-bin curves, RMS difference <= 0.08
  v   spatial contrast spectrum (band-pass std / mean, sigma = column width / 256 .. / 4), RMS log2 ratio <= 0.5
  vi  bright coherence: largest bright (>=p70) component share of bright pixels within +-0.15,
      bright fragment count per 1000 bright px ratio within [0.5, 2]
  vii interior (deeper than width/8 from the silhouette edge) dark (<80) fraction, diff <= +0.05
  viii puff isotropy (vertical / horizontal 1/e correlation length of the detail) ratio within [0.7, 1.4],
      puff scale (mean correlation length / width) ratio within [0.6, 1.6]
  ix  centerline bend: RMS deviation of the row centre from a straight-line fit over bands 1..6 (of 0..9,
      above the base pool) / width, ratio within [0.5, 2]; per-band centre std of bands 2..6 ratio within [0.5, 2]
  x   width p90/median per band ratio within [0.8, 1.25]; widest row of the bottom 15% / middle median width
      ratio within [0.7, 1.4]
  xi  detached silhouette fragments (area >= (width/16)^2) in the top quarter: count diff within +-1,
      median width ratio within [0.5, 2] (only gated when the reference has fragments)
  xvii dim halo (the flame outside the silhouette but above R>35 && R-B>15, rows above the base pool):
      extra width (halo width - silhouette width) / column width ratio within [0.7, 1.4]; tongues = halo
      runs at least width/4 tall protruding > 0.25 width beyond the silhouette on one side, count ratio
      within [0.5, 2]
  xviii video gate: see flame_ref_match_video.py (--video, implies --temporal); whole-clip streaming: moving
      pair fraction, upward transport of the width and centre fields, centre phase lag between bands,
      centre return. The summary line reports the video score (mean of the xviii scores) separately.
  xiii-xvi sequence gates: see flame_ref_match_sequence.py (--temporal; the render must be a sequence at
      --fps). The reference frame rate is read from the reference meta.json; --ref-window narrows the
      reference span (default: every frame). Frames listed as caption_frames in meta.json carry burned-in
      text and are excluded from the static reference (the silhouette fields the sequence gates use are
      hole-filled, so captions inside the column do not disturb them).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import flame_ref_match_sequence as sequence
import flame_ref_match_video as video

REF_DIR = Path(__file__).resolve().parents[1] / "assets/textures/flames/pillar_ref_seq_stable"
LUM_BINS = [(40, 80), (80, 120), (120, 160), (160, 200), (200, 256)]
BAND_COUNT = 10
ROOT_BAND = 7
DARK_LUM = 80.0
CONTRAST_SIGMA_WIDTH_FRACTIONS = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]
BRIGHT_PERCENTILE = 70.0
HALO_TONGUE_PROTRUSION = 0.25
HALO_BASE_POOL_FRACTION = 0.2
def load_ref_meta(ref_dir):
    meta_path = Path(ref_dir) / "meta.json"
    if not meta_path.exists():
        return 10.0, set()
    meta = json.loads(meta_path.read_text())
    return float(meta.get("fps", 10.0)), set(meta.get("caption_frames", []))


def load_image(path, crop=None):
    image = Image.open(path).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    return image


def to_rgb(image):
    return np.asarray(image).astype(np.float64)


def median_column_width(rgb):
    widths = silhouette_mask(rgb).sum(axis=1)
    return float(np.median(widths[widths > 0])) if (widths > 0).any() else 0.0


def resample_to_column_width(image, target_width):
    width = median_column_width(to_rgb(image))
    if width <= 0.0:
        return image
    scale = target_width / width
    if scale > 1.01:
        print(f"warning: render column width {width:.0f} px is below the reference {target_width:.0f} px; "
              "upscaling loses the fine detail the gates measure", file=sys.stderr)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    return image.resize(size, resample)


SILHOUETTE_MODES = ("flame", "dust")
silhouette_mode = "flame"
DUST_SILHOUETTE_LUM = 60.0
DUST_HALO_LUM = 20.0


def silhouette_mask(rgb):
    if silhouette_mode == "dust":
        return luminance(rgb) > DUST_SILHOUETTE_LUM
    r, b = rgb[:, :, 0], rgb[:, :, 2]
    return (r > 90) & (r - b > 40)


def halo_mask(rgb):
    if silhouette_mode == "dust":
        return luminance(rgb) > DUST_HALO_LUM
    r, b = rgb[:, :, 0], rgb[:, :, 2]
    return (r > 35) & (r - b > 15)


def row_edges(mask):
    first = np.where(mask.any(axis=1), mask.argmax(axis=1), np.nan)
    last = np.where(mask.any(axis=1), mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1), np.nan)
    return first.astype(float), last.astype(float)


def halo_spread(rgb, mask, column_width):
    """(extra width / column width, tongue count) of the dim halo above the base pool."""
    core = ndimage.binary_fill_holes(mask)
    halo = ndimage.binary_fill_holes(halo_mask(rgb) | mask)
    ys = np.where(core.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max() + 1
    cut = y0 + int((y1 - y0) * (1.0 - HALO_BASE_POOL_FRACTION))
    core, halo = core[y0:cut], halo[y0:cut]
    rows = core.any(axis=1)
    if rows.sum() < 10:
        return np.nan, np.nan

    extra_width = float(((halo.sum(axis=1) - core.sum(axis=1))[rows]).mean() / column_width)
    core_left, core_right = row_edges(core)
    halo_left, halo_right = row_edges(halo)
    protrusion = np.nan_to_num(np.maximum(halo_right - core_right, core_left - halo_left)) / column_width
    runs, count = ndimage.label(protrusion > HALO_TONGUE_PROTRUSION)
    run_heights = ndimage.sum(np.ones_like(runs), runs, range(1, count + 1)) if count else []
    tongues = float(sum(1 for height in run_heights if height >= column_width / 4.0))
    return extra_width, tongues


def luminance(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def contrast_spectrum(lum, mask, column_width):
    erosion = max(1, round(column_width / 32))
    inner = ndimage.binary_erosion(mask, iterations=erosion)
    if inner.sum() < 50:
        return [np.nan] * len(CONTRAST_SIGMA_WIDTH_FRACTIONS)
    filled = np.where(mask, lum, lum[mask].mean())
    mean_lum = lum[inner].mean()
    spectrum = []
    for fraction in CONTRAST_SIGMA_WIDTH_FRACTIONS:
        sigma = fraction * column_width
        band = ndimage.gaussian_filter(filled, sigma * 0.5) - ndimage.gaussian_filter(filled, sigma)
        spectrum.append(float(band[inner].std() / mean_lum))
    return spectrum


def bright_coherence(lum, mask):
    lum_n, mask_n = lum, mask
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


def interior_hole_ratio(lum, mask, column_width):
    filled = ndimage.binary_fill_holes(mask)
    distance = ndimage.distance_transform_edt(filled)
    interior = distance > column_width / 8.0
    if interior.sum() < 50:
        return np.nan
    return float((lum[interior] < DARK_LUM).mean())


def correlation_length(signal, valid, axis, max_lag):
    lags = np.arange(1, max_lag + 1)
    correlations = []
    variance = signal[valid].var()
    for lag in lags:
        a = np.take(signal, range(signal.shape[axis] - lag), axis=axis)
        b = np.take(signal, range(lag, signal.shape[axis]), axis=axis)
        va = np.take(valid, range(valid.shape[axis] - lag), axis=axis)
        vb = np.take(valid, range(lag, valid.shape[axis]), axis=axis)
        both = va & vb
        if both.sum() < 50 or variance <= 0:
            return np.nan
        correlations.append((a[both] * b[both]).mean() / variance)
    correlations = np.array(correlations)
    below = np.where(correlations < np.exp(-1.0))[0]
    if below.size == 0:
        return float(max_lag)
    k = below[0]
    if k == 0:
        return float(lags[0] * correlations[0])
    c0, c1 = correlations[k - 1], correlations[k]
    return float(lags[k - 1] + (c0 - np.exp(-1.0)) / max(c0 - c1, 1e-6))


def puff_isotropy(lum, mask, column_width):
    interior = ndimage.binary_erosion(mask, iterations=max(1, round(column_width / 32)))
    if interior.sum() < 200:
        return np.nan, np.nan
    filled = np.where(mask, lum, lum[mask].mean())
    detail = filled - ndimage.gaussian_filter(filled, column_width / 4.0)
    detail = detail - detail[interior].mean()
    max_lag = max(2, round(column_width / 2))
    length_x = correlation_length(detail, interior, 1, max_lag)
    length_y = correlation_length(detail, interior, 0, max_lag)
    if np.isnan(length_x) or np.isnan(length_y) or length_x <= 0:
        return np.nan, np.nan
    return float(length_y / length_x), float((length_x + length_y) / 2.0 / column_width)


def vertical_modulation(lum, mask, column_width):
    rows = np.where(mask.any(axis=1))[0]
    y0, y1 = rows.min(), rows.max() + 1
    row_counts = mask[y0:y1].sum(axis=1)
    row_means = (lum[y0:y1] * mask[y0:y1]).sum(axis=1) / np.maximum(row_counts, 1)
    profile = row_means[row_counts > 0.3 * column_width]
    if profile.size < column_width:
        return [np.nan, np.nan]
    out = []
    for sigma in (column_width / 8.0, column_width / 4.0):
        band = ndimage.gaussian_filter1d(profile, sigma) - ndimage.gaussian_filter1d(profile, 2.0 * sigma)
        out.append(float(band.std() / profile.mean()))
    return out


def row_bands(mask):
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max() + 1
    edges = [y0 + (y1 - y0) * band // BAND_COUNT for band in range(BAND_COUNT + 1)]
    return list(zip(edges[:-1], edges[1:]))


def row_centers(mask):
    xs = np.arange(mask.shape[1])
    counts = mask.sum(axis=1)
    return np.where(counts > 0, (mask * xs).sum(axis=1) / np.maximum(counts, 1), np.nan)


def centerline_amplitude(mask, column_width):
    centers = row_centers(mask)
    amplitudes = []
    for ya, yb in row_bands(mask):
        band = centers[ya:yb]
        band = band[~np.isnan(band)]
        amplitudes.append(float(band.std() / column_width) if band.size >= 3 else np.nan)
    return amplitudes


def centerline_straightness(mask, column_width):
    centers = row_centers(mask)
    bands = row_bands(mask)
    ya, yb = bands[1][0], bands[ROOT_BAND][0]
    rows = np.arange(ya, yb)
    values = centers[ya:yb]
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return np.nan
    slope, intercept = np.polyfit(rows[valid], values[valid], 1)
    deviation = values[valid] - (slope * rows[valid] + intercept)
    return float(np.sqrt((deviation ** 2).mean()) / column_width)


def width_profile(mask, column_width):
    widths = mask.sum(axis=1).astype(float)
    spread = []
    for ya, yb in row_bands(mask):
        band = widths[ya:yb]
        band = band[band > 0]
        spread.append(float(np.percentile(band, 90) / np.median(band)) if band.size >= 3 else np.nan)
    ys = np.where(widths > 0)[0]
    y0, y1 = ys.min(), ys.max() + 1
    rows = y1 - y0
    base = widths[y1 - max(1, int(rows * 0.15)):y1]
    middle = widths[y0 + int(rows * 0.3):y0 + int(rows * 0.7)]
    base_ratio = float(base.max() / np.median(middle[middle > 0]))
    return spread, base_ratio


def top_fragments(lum, mask, column_width):
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max() + 1
    cut = y0 + (y1 - y0) // 4
    labels, count = ndimage.label(mask)
    attached = set(np.unique(labels[cut - 1:cut + 1])) - {0}
    min_area = (column_width / 16.0) ** 2
    widths = []
    for label in range(1, count + 1):
        if label in attached:
            continue
        rows, cols = np.where(labels[:cut] == label)
        if rows.size < min_area:
            continue
        widths.append((cols.max() - cols.min() + 1) / column_width)
    median_width = float(np.median(widths)) if widths else 0.0
    return float(len(widths)), median_width


def frame_stats(rgb, column_width):
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
        "contrast_spectrum": contrast_spectrum(lum, mask, column_width),
        "bright_largest_share": largest_share,
        "bright_fragments_per_k": fragments_per_k,
        "interior_hole_ratio": interior_hole_ratio(lum, mask, column_width),
        "puff_isotropy_scale": list(puff_isotropy(lum, mask, column_width)),
        "halo_spread": list(halo_spread(rgb, mask, column_width)),
        "centerline_amplitude": centerline_amplitude(mask, column_width),
        "centerline_straightness": centerline_straightness(mask, column_width),
        "width_spread_base": width_profile(mask, column_width),
        "top_fragments": list(top_fragments(lum, mask, column_width)),
        "vertical_modulation": vertical_modulation(lum, mask, column_width),
    }


def aggregate(stat_list):
    keys = ["lum_p10_50_90", "dark_fraction", "bright_fraction", "band_mean", "band_p90", "gr_by_lum", "br_by_lum",
            "contrast_spectrum", "bright_largest_share", "bright_fragments_per_k", "interior_hole_ratio",
            "puff_isotropy_scale", "centerline_amplitude", "centerline_straightness", "top_fragments",
            "vertical_modulation", "halo_spread"]
    out = {"frames": len(stat_list)}
    for key in keys:
        out[key] = np.nanmean(np.array([s[key] for s in stat_list], dtype=float), axis=0).tolist()
    out["width_spread"] = np.nanmean(np.array([s["width_spread_base"][0] for s in stat_list]), axis=0).tolist()
    out["base_width_ratio"] = float(np.nanmean([s["width_spread_base"][1] for s in stat_list]))
    return out


def collect_frames(target):
    target = Path(target)
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() == ".png")
    return [target]


def reference_column_width(paths):
    widths = [median_column_width(to_rgb(load_image(p))) for p in paths]
    return float(np.median([w for w in widths if w > 0]))


def measure(paths, column_width, crop=None, resample=True):
    stats = []
    for path in paths:
        image = load_image(path, crop)
        if resample:
            image = resample_to_column_width(image, column_width)
        stat = frame_stats(to_rgb(image), column_width)
        if stat is not None:
            stats.append(stat)
    if not stats:
        sys.exit(f"no flame silhouette found in {paths[0]} ...")
    aggregated = aggregate(stats)
    aggregated["column_width_px"] = column_width
    return aggregated


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
    rows.extend(compare_structure(ref, render))
    return rows


def gate_score(name, value, rv, cv):
    """Similarity in [0, 1] of one gate row, 1 = identical to the reference.

    ratio rows: min(r, 1/r); diff rows: 1 - |render - ref| / max(|ref|, |render|); correlation rows: the
    correlation; rms rows: 1 - rms (log2 rms rows: 1 - rms / 2, so a factor 4 scores 0)."""
    kind, _, number = value.partition(" ")
    if kind == "ratio":
        r = float(number)
        return float(min(r, 1.0 / r)) if r > 0 else 0.0
    if kind == "diff":
        a, b = float(rv), float(cv)
        scale = max(abs(a), abs(b))
        return 1.0 if scale == 0 else float(max(0.0, 1.0 - abs(b - a) / scale))
    x = float(kind)
    if "log2" in name:
        return float(max(0.0, 1.0 - x / 2.0))
    if "rms" in name:
        return float(max(0.0, 1.0 - x))
    return float(max(0.0, min(1.0, x)))


def score_rows(rows):
    return [(name, rv, cv, value, ok, gate_score(name, value, rv, cv)) for name, rv, cv, value, ok in rows]


def ratio_row(label, rv, cv, low, high, fmt=".2f"):
    ratio = cv / rv if rv else np.nan
    return (label, f"{rv:{fmt}}", f"{cv:{fmt}}", f"ratio {ratio:.2f}", bool(low <= ratio <= high))


def compare_structure(ref, render):
    rows = []
    diff = render["interior_hole_ratio"] - ref["interior_hole_ratio"]
    rows.append(("vii interior hole", f"{ref['interior_hole_ratio']:.3f}", f"{render['interior_hole_ratio']:.3f}",
                 f"diff {diff:+.3f}", diff <= 0.05))
    rows.append(ratio_row("viii puff isotropy", ref["puff_isotropy_scale"][0], render["puff_isotropy_scale"][0], 0.7, 1.4))
    rows.append(ratio_row("viii puff scale", ref["puff_isotropy_scale"][1], render["puff_isotropy_scale"][1], 0.6, 1.6, ".3f"))
    rows.append(ratio_row("ix  centerline bend", ref["centerline_straightness"], render["centerline_straightness"], 0.5, 2.0, ".3f"))
    middle_ref = float(np.nanmean(ref["centerline_amplitude"][2:ROOT_BAND]))
    middle_render = float(np.nanmean(render["centerline_amplitude"][2:ROOT_BAND]))
    rows.append(ratio_row("ix  middle amplitude", middle_ref, middle_render, 0.5, 2.0, ".3f"))
    rows.append(ratio_row("x   width p90/median", float(np.nanmean(ref["width_spread"])), float(np.nanmean(render["width_spread"])), 0.8, 1.25))
    rows.append(ratio_row("x   base width ratio", ref["base_width_ratio"], render["base_width_ratio"], 0.7, 1.4))
    count_ref, count_render = ref["top_fragments"][0], render["top_fragments"][0]
    diff = count_render - count_ref
    rows.append(("xi  top fragments", f"{count_ref:.1f}", f"{count_render:.1f}", f"diff {diff:+.1f}", abs(diff) <= 1.0))
    width_row = ratio_row("xi  top fragment width", ref["top_fragments"][1], render["top_fragments"][1], 0.5, 2.0, ".3f")
    if count_ref < 0.5:
        width_row = width_row[:4] + (None,)
    rows.append(width_row)
    rows.append(ratio_row("xii vertical mod w/8", ref["vertical_modulation"][0], render["vertical_modulation"][0], 0.6, 1.6, ".3f"))
    rows.append(ratio_row("xii vertical mod w/4", ref["vertical_modulation"][1], render["vertical_modulation"][1], 0.6, 1.6, ".3f"))
    rows.append(ratio_row("xvii halo extra width", ref["halo_spread"][0], render["halo_spread"][0], 0.7, 1.4))
    rows.append(ratio_row("xvii halo tongues", ref["halo_spread"][1], render["halo_spread"][1], 0.5, 2.0, ".1f"))
    return rows


def print_profiles(ref, render):
    print("\nheight bands (top -> bottom), lum mean / p90:")
    for band in range(BAND_COUNT):
        rm, rp = ref["band_mean"][band], ref["band_p90"][band]
        cm, cp = render["band_mean"][band], render["band_p90"][band]
        print(f"  band {band}: ref {rm:5.0f}/{rp:5.0f}   render {cm:5.0f}/{cp:5.0f}")
    width = ref["column_width_px"]
    print(f"\ncontrast spectrum (band-pass std / mean, sigma in px at column width {width:.0f}):")
    for fraction, rv, cv in zip(CONTRAST_SIGMA_WIDTH_FRACTIONS, ref["contrast_spectrum"], render["contrast_spectrum"]):
        print(f"  sigma {fraction * width:5.1f} (1/{1 / fraction:.0f} width): ref {rv:.3f} render {cv:.3f}  x{cv / rv:.2f}")
    print(f"\ncenterline bend (rms from line / width): ref {ref['centerline_straightness']:.3f} render {render['centerline_straightness']:.3f}")
    print("centerline amplitude / width p90/median by band (top -> bottom):")
    for band in range(BAND_COUNT):
        print(f"  band {band}: amp ref {ref['centerline_amplitude'][band]:.3f} render {render['centerline_amplitude'][band]:.3f}"
              f"   spread ref {ref['width_spread'][band]:.2f} render {render['width_spread'][band]:.2f}")
    print("\nG/R and B/R by lum bin:")
    for (lo, hi), rg, cg, rb, cb in zip(LUM_BINS, ref["gr_by_lum"], render["gr_by_lum"], ref["br_by_lum"], render["br_by_lum"]):
        print(f"  lum {lo:3d}-{hi:3d}: G/R ref {rg:.2f} render {cg:.2f} | B/R ref {rb:.2f} render {cb:.2f}")


def measure_sequence(paths, column_width, dt, crop=None, resample=True, with_video=False):
    rgbs = sequence.load_sequence(paths, column_width, crop, resample, load_image, to_rgb, median_column_width)
    temporal = sequence.measure_temporal(rgbs, column_width, dt, silhouette_mask, row_centers)
    if with_video:
        temporal["video"] = video.measure_video(rgbs, column_width, dt, silhouette_mask, row_centers)
    return temporal


def print_summary(rows):
    passed = sum(1 for r in rows if r[4])
    gated = sum(1 for r in rows if r[4] is not None)
    scores = [r[5] for r in rows if r[4] is not None]
    above = sum(1 for v in scores if v >= 0.95)
    print(f"\n{passed}/{gated} gates pass, score mean {np.mean(scores):.3f} min {min(scores):.2f}, {above}/{gated} >= 0.95")
    video_rows = [r for r in rows if r[0].startswith("xviii")]
    if video_rows:
        video_scores = [r[5] for r in video_rows]
        video_passed = sum(1 for r in video_rows if r[4])
        print(f"video score {np.mean(video_scores):.3f} ({video_passed}/{len(video_rows)} xviii gates pass)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", help="rendered png or directory of pngs")
    parser.add_argument("--crop", help="x0,y0,x1,y1 crop applied to every rendered png")
    parser.add_argument("--ref-dir", default=str(REF_DIR))
    parser.add_argument("--ref-only", action="store_true")
    parser.add_argument("--json", help="write reference/render/gate results to this path")
    parser.add_argument("--temporal", action="store_true", help="also measure the sequence gates xiii-xvi")
    parser.add_argument("--video", action="store_true", help="also measure the whole-clip video gate xviii (implies --temporal)")
    parser.add_argument("--fps", type=float, default=10.0, help="frame rate of the render sequence")
    parser.add_argument("--silhouette", choices=SILHOUETTE_MODES, default="flame",
                        help="silhouette definition: flame colours, or dust (luminance on a black background)")
    parser.add_argument("--ref-window", default=None,
                        help="first,end reference frame for the sequence gates (default: every frame)")
    args = parser.parse_args()
    args.temporal = args.temporal or args.video
    global silhouette_mode
    silhouette_mode = args.silhouette

    ref_fps, caption_frames = load_ref_meta(args.ref_dir)
    ref_paths = collect_frames(args.ref_dir)
    static_paths = [p for i, p in enumerate(ref_paths) if i not in caption_frames]
    column_width = reference_column_width(static_paths)
    ref = measure(static_paths, column_width, resample=False)
    print(f"reference: {ref['frames']} frames, column width {column_width:.0f} px, lum p10/50/90 = "
          f"{ref['lum_p10_50_90'][0]:.0f}/{ref['lum_p10_50_90'][1]:.0f}/{ref['lum_p10_50_90'][2]:.0f}, "
          f"dark {ref['dark_fraction'] * 100:.0f}%, bright {ref['bright_fraction'] * 100:.0f}%")
    ref_temporal = None
    if args.temporal:
        first, end = (int(v) for v in args.ref_window.split(",")) if args.ref_window else (0, len(ref_paths))
        ref_temporal = measure_sequence(ref_paths[first:end], column_width, 1.0 / ref_fps, resample=False,
                                        with_video=args.video)
    if args.ref_only or not args.render:
        if ref_temporal is not None:
            sequence.print_temporal(ref_temporal, ref_temporal)
            if args.video:
                video.print_video(ref_temporal["video"], ref_temporal["video"])
            if args.json:
                Path(args.json).write_text(json.dumps({"reference_temporal": ref_temporal}, indent=2))
        return

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    render = measure(collect_frames(args.render), column_width, crop)
    print(f"render:    {render['frames']} frames, lum p10/50/90 = "
          f"{render['lum_p10_50_90'][0]:.0f}/{render['lum_p10_50_90'][1]:.0f}/{render['lum_p10_50_90'][2]:.0f}, "
          f"dark {render['dark_fraction'] * 100:.0f}%, bright {render['bright_fraction'] * 100:.0f}%")

    rows = compare(ref, render)
    render_temporal = None
    if args.temporal:
        render_temporal = measure_sequence(collect_frames(args.render), column_width, 1.0 / args.fps, crop,
                                           with_video=args.video)
        rows.extend(sequence.compare_temporal(ref_temporal, render_temporal, ratio_row))
        if args.video:
            rows.extend(video.compare_video(ref_temporal["video"], render_temporal["video"]))
    rows = score_rows(rows)
    print(f"\n{'gate':<26}{'ref':>6}{'render':>8}  {'value':<14}{'status':<6}score")
    for name, rv, cv, value, ok, score in rows:
        status = "-" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{name:<26}{rv:>6}{cv:>8}  {value:<14}{status:<6}{score:.2f}")
    print_summary(rows)
    print_profiles(ref, render)
    if render_temporal is not None:
        sequence.print_temporal(ref_temporal, render_temporal)
        if args.video:
            video.print_video(ref_temporal["video"], render_temporal["video"])

    if args.json:
        Path(args.json).write_text(json.dumps({"reference": ref, "render": render,
                                               "reference_temporal": ref_temporal, "render_temporal": render_temporal,
                                               "gates": [{"name": r[0], "value": r[3], "pass": r[4], "score": r[5]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
