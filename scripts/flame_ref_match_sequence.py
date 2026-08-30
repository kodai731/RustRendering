"""Sequence gates xiii-xvi for flame_ref_match.py (frame sequences at a fixed frame interval).

Time is playback time: the reference video is slow motion, so the gates compare what a viewer sees when
the frames play at 10 fps against the render advancing at its own frame rate.

The reference column is saturated inside (lum p90 ~ 200), so the puffs a viewer sees are lobes of the
silhouette, not luminance lumps. The sequence gates therefore work on two 1D fields over height and
time: the row width w(y, t) and the row centre c(y, t) of the silhouette, both in column widths.
Every frame is resampled with one common scale (median over the frames of the column width).

  xvi rise speed: per band (4 height bands, top -> bottom), vertical shift maximising the correlation of
      the width detail (band-pass sigma width/8 .. width) between consecutive frames, in column widths
      per second; difference within +-0.2 (the reference lobes barely rise in playback time)
  xiii change energy: RMS of the width change between consecutive frames / width std over height, raw
      and after the per-band rise shift (residual = change a pure upward translation cannot explain);
      ratio within [0.6, 1.6]
  xiv lobe lifetime: width-detail maxima (prominence >= 0.5 std) tracked across frames to the nearest
      maximum within width/4 after the rise shift; median chain length (frames) ratio within [0.5, 2],
      (births + deaths) per frame per mean lobe count difference within +-0.15
  xv centreline spectrum: per band, power spectrum of the band centre x(t) (Hann window) on a fixed
      0.25 .. 5 Hz grid; RMS of log2 power ratio <= 0.6, dominant frequency within +-1 Hz
"""

import numpy as np
from scipy import ndimage, signal

TEMPORAL_BAND_COUNT = 4
WIDTH_DETAIL_SIGMA_FRACTIONS = (1 / 8, 1.0)
LOBE_PROMINENCE_STD = 0.5
RISE_MIN_CORRELATION = 0.3
SPECTRUM_HZ = np.arange(0.25, 5.01, 0.25)
RISE_TOLERANCE = 0.2
TURNOVER_TOLERANCE = 0.15


def load_sequence(paths, column_width, crop, resample, load_image, to_rgb, median_column_width):
    from PIL import Image

    images = [load_image(path, crop) for path in paths]
    if resample:
        widths = [median_column_width(to_rgb(image)) for image in images]
        widths = [w for w in widths if w > 0]
        scale = column_width / float(np.median(widths)) if widths else 1.0
        method = Image.BOX if scale < 1.0 else Image.BICUBIC
        images = [image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), method)
                  for image in images]
    return [to_rgb(image) for image in images]


def row_extent(masks):
    union = np.logical_or.reduce(masks)
    ys = np.where(union.any(axis=1))[0]
    return int(ys.min()), int(ys.max()) + 1


def temporal_bands(row_count):
    edges = [row_count * band // TEMPORAL_BAND_COUNT for band in range(TEMPORAL_BAND_COUNT + 1)]
    return list(zip(edges[:-1], edges[1:]))


def silhouette_profiles(masks, column_width, row_centers):
    """Width and centre per row (column widths), rows = union extent; nan where the row is empty."""
    y0, y1 = row_extent(masks)
    widths, centres = [], []
    for mask in masks:
        filled = ndimage.binary_fill_holes(mask)[y0:y1]
        count = filled.sum(axis=1).astype(float)
        widths.append(np.where(count > 0, count, np.nan) / column_width)
        centres.append(row_centers(mask)[y0:y1] / column_width)
    return np.array(widths), np.array(centres)


def fill_nan_rows(profile):
    valid = ~np.isnan(profile)
    if valid.sum() < 2:
        return np.zeros_like(profile)
    rows = np.arange(profile.size)
    return np.interp(rows, rows[valid], profile[valid])


def width_detail(widths, column_width):
    fine, coarse = (fraction * column_width for fraction in WIDTH_DETAIL_SIGMA_FRACTIONS)
    return np.array([ndimage.gaussian_filter1d(fill_nan_rows(w), fine) - ndimage.gaussian_filter1d(fill_nan_rows(w), coarse)
                     for w in widths])


def best_vertical_shift(before, after, rows, max_shift):
    ya, yb = rows
    length = before.size
    best_shift, best_corr = 0, -np.inf
    for shift in range(-max_shift, max_shift + 1):
        src_a, src_b = max(ya + shift, 0), min(yb + shift, length)
        if src_b - src_a < 8:
            continue
        a, b = before[src_a:src_b], after[src_a - shift:src_b - shift]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_shift, best_corr = shift, corr
    return best_shift, best_corr


def pair_shifts(details, bands, column_width):
    """Per frame pair and band: rows to move frame t up by to land on frame t+1 (> 0 is rising)."""
    max_shift = max(1, round(column_width / 2))
    shifts = np.full((details.shape[0] - 1, len(bands)), np.nan)
    for index in range(details.shape[0] - 1):
        for band, rows in enumerate(bands):
            shift, corr = best_vertical_shift(details[index], details[index + 1], rows, max_shift)
            if corr >= RISE_MIN_CORRELATION:
                shifts[index, band] = shift
    return shifts


def rise_speed(shifts, column_width, dt):
    return [float(np.nanmedian(shifts[:, band]) / column_width / dt) if np.isfinite(shifts[:, band]).any() else np.nan
            for band in range(shifts.shape[1])]


def shift_profile_by_band(profile, bands, band_shifts):
    shifted = np.full_like(profile, np.nan)
    length = profile.size
    for (ya, yb), shift in zip(bands, band_shifts):
        shift = int(shift) if np.isfinite(shift) else 0
        src_a, src_b = max(ya + shift, 0), min(yb + shift, length)
        shifted[src_a - shift:src_b - shift] = profile[src_a:src_b]
    return shifted


def change_energy(widths, bands, shifts):
    raw, residual = [], []
    for index in range(widths.shape[0] - 1):
        before, after = widths[index], widths[index + 1]
        scale = 0.5 * (np.nanstd(before) + np.nanstd(after))
        if not np.isfinite(scale) or scale < 1e-9:
            continue
        raw.append(float(np.sqrt(np.nanmean((after - before) ** 2)) / scale))
        aligned = shift_profile_by_band(before, bands, shifts[index])
        residual.append(float(np.sqrt(np.nanmean((after - aligned) ** 2)) / scale))
    return [float(np.mean(raw)) if raw else np.nan, float(np.mean(residual)) if residual else np.nan]


def lobe_rows(detail):
    prominence = LOBE_PROMINENCE_STD * detail.std()
    if prominence <= 0:
        return np.array([], dtype=int)
    peaks, _ = signal.find_peaks(detail, prominence=prominence)
    return peaks


def row_band(row, bands):
    for band, (ya, yb) in enumerate(bands):
        if ya <= row < yb:
            return band
    return len(bands) - 1


def lobe_lifetimes(details, bands, shifts, column_width):
    lobes = [lobe_rows(detail) for detail in details]
    chain_length = {int(row): 1 for row in lobes[0]}
    finished, births, deaths = [], 0, 0
    reach = column_width / 4.0
    for index in range(details.shape[0] - 1):
        next_chain = {}
        taken = set()
        for row, length in chain_length.items():
            shift = shifts[index, row_band(row, bands)]
            expected = row - (int(shift) if np.isfinite(shift) else 0)
            candidates = [int(r) for r in lobes[index + 1] if abs(r - expected) <= reach and int(r) not in taken]
            if candidates:
                nearest = min(candidates, key=lambda r: abs(r - expected))
                next_chain[nearest] = length + 1
                taken.add(nearest)
            else:
                finished.append(length)
                deaths += 1
        for row in lobes[index + 1]:
            if int(row) not in next_chain:
                next_chain[int(row)] = 1
                births += 1
        chain_length = next_chain
    finished.extend(chain_length.values())
    mean_count = float(np.mean([len(rows) for rows in lobes]))
    if not finished or mean_count <= 0:
        return [np.nan, np.nan]
    return [float(np.median(finished)), float((births + deaths) / (details.shape[0] - 1) / mean_count)]


def band_centre_series(centres, bands):
    return np.array([np.nanmean(centres[:, ya:yb], axis=1) for ya, yb in bands])


def centreline_spectrum(series, dt):
    power_grid, dominant = [], []
    for sample in series:
        sample = np.where(np.isnan(sample), np.nanmean(sample), sample)
        sample = (sample - sample.mean()) * np.hanning(sample.size)
        power = np.abs(np.fft.rfft(sample)) ** 2 / sample.size
        freqs = np.fft.rfftfreq(sample.size, dt)
        power_grid.append(np.interp(SPECTRUM_HZ, freqs, np.log2(power + 1e-12)).tolist())
        positive = freqs > 0
        dominant.append(float(freqs[positive][np.argmax(power[positive])]))
    return power_grid, dominant


def measure_temporal(rgbs, column_width, dt, silhouette_mask, row_centers):
    masks = [silhouette_mask(rgb) for rgb in rgbs]
    widths, centres = silhouette_profiles(masks, column_width, row_centers)
    details = width_detail(widths, column_width)
    bands = temporal_bands(widths.shape[1])

    shifts = pair_shifts(details, bands, column_width)
    spectrum, dominant = centreline_spectrum(band_centre_series(centres, bands), dt)
    return {
        "frames": len(rgbs),
        "dt": dt,
        "rise_speed": rise_speed(shifts, column_width, dt),
        "change_energy": change_energy(widths, bands, shifts),
        "lobe_lifetime_turnover": lobe_lifetimes(details, bands, shifts, column_width),
        "centreline_log2_power": spectrum,
        "centreline_dominant_hz": dominant,
    }


def diff_row(label, rv, cv, tolerance):
    diff = cv - rv
    return (label, f"{rv:.2f}", f"{cv:.2f}", f"diff {diff:+.2f}", bool(np.isfinite(diff) and abs(diff) <= tolerance))


def compare_temporal(ref, render, ratio_row):
    rows = []
    for band in range(TEMPORAL_BAND_COUNT):
        rows.append(diff_row(f"xvi rise speed b{band}", ref["rise_speed"][band], render["rise_speed"][band], RISE_TOLERANCE))
    rows.append(ratio_row("xiii change raw", ref["change_energy"][0], render["change_energy"][0], 0.6, 1.6, ".3f"))
    rows.append(ratio_row("xiii change residual", ref["change_energy"][1], render["change_energy"][1], 0.6, 1.6, ".3f"))
    rows.append(ratio_row("xiv lobe lifetime", ref["lobe_lifetime_turnover"][0], render["lobe_lifetime_turnover"][0], 0.5, 2.0, ".1f"))
    rows.append(diff_row("xiv lobe turnover", ref["lobe_lifetime_turnover"][1], render["lobe_lifetime_turnover"][1], TURNOVER_TOLERANCE))
    log_ratio = np.array(render["centreline_log2_power"]) - np.array(ref["centreline_log2_power"])
    rms = float(np.sqrt(np.nanmean(log_ratio ** 2)))
    rows.append(("xv  centreline log2 rms", "", "", f"{rms:.2f}", rms <= 0.6))
    for band in range(TEMPORAL_BAND_COUNT):
        rf, cf = ref["centreline_dominant_hz"][band], render["centreline_dominant_hz"][band]
        rows.append((f"xv  dominant hz b{band}", f"{rf:.2f}", f"{cf:.2f}", f"diff {cf - rf:+.2f}", abs(cf - rf) <= 1.0))
    return rows


def print_temporal(ref, render):
    print("\ntemporal (bands top -> bottom): rise speed [width/s], centreline dominant hz")
    for band in range(TEMPORAL_BAND_COUNT):
        print(f"  band {band}: rise ref {ref['rise_speed'][band]:.2f} render {render['rise_speed'][band]:.2f}"
              f"   hz ref {ref['centreline_dominant_hz'][band]:.2f} render {render['centreline_dominant_hz'][band]:.2f}")
    print(f"width change raw/residual: ref {ref['change_energy'][0]:.3f}/{ref['change_energy'][1]:.3f}"
          f" render {render['change_energy'][0]:.3f}/{render['change_energy'][1]:.3f}")
    print(f"lobe lifetime [frames] / turnover: ref {ref['lobe_lifetime_turnover'][0]:.1f}/{ref['lobe_lifetime_turnover'][1]:.2f}"
          f" render {render['lobe_lifetime_turnover'][0]:.1f}/{render['lobe_lifetime_turnover'][1]:.2f}")
