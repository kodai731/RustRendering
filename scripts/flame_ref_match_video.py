"""Video gate xviii for flame_ref_match.py: does the flame stream as a whole, the way the reference does?

The sequence gates xiii-xvi are per-band statistics of one field each; this gate asks the whole-clip
question a viewer asks, "is the motion carried upward or does it happen in place?", on both silhouette
fields, the row width w(y, t) and the row centre c(y, t), in column widths after removing each row's mean.

  xviii moving pairs: per band, fraction of consecutive frame pairs whose width detail moved by at least
        one row (correlation >= RISE_MIN_CORRELATION); a streaming flame moves in every pair, a flame that
        only slides sideways and settles back does not; difference within +-0.25
  xviii transport: over the whole column and TRANSPORT_LAG_SECONDS, the upward shift (widths/s) that best
        aligns the field with its later self, for the width detail and the centre field, per band; a
        shift counts only when it raises the correlation by TRANSPORT_MIN_GAIN over no shift (else the
        motion happens in place, speed 0); difference within +-0.2; the gains are printed for information
  xviii centre phase lag: lag (s) of the best cross-correlation between the centre series of a band and
        the band above it; transported lateral motion shows a positive lag, in-phase sliding shows 0;
        difference within +-0.1 s
  xviii centre return: minimum autocorrelation of each band's detrended centre series over lags up to
        half the clip; a displacement carried away decorrelates (-> 0), one pulled back to the same place
        anti-correlates (-> -1); difference within +-0.25
"""

import numpy as np

from flame_ref_match_sequence import (TEMPORAL_BAND_COUNT, band_centre_series, fill_nan_rows, pair_shifts,
                                      silhouette_profiles, temporal_bands, width_detail)

TRANSPORT_LAG_SECONDS = 0.1
TRANSPORT_MAX_SHIFT_WIDTHS = 1.5
TRANSPORT_MIN_GAIN = 0.01
PHASE_MAX_LAG_SECONDS = 0.5
MOVING_PAIR_TOLERANCE = 0.25
TRANSPORT_SPEED_TOLERANCE = 0.2
PHASE_LAG_TOLERANCE = 0.1
RETURN_TOLERANCE = 0.25


def moving_pair_fraction(shifts):
    return [float(np.mean(np.abs(shifts[:, band]) >= 1)) if np.isfinite(shifts[:, band]).any() else np.nan
            for band in range(shifts.shape[1])]


def centre_detail(centres):
    filled = np.array([fill_nan_rows(row) for row in centres])
    return filled - filled.mean(axis=0, keepdims=True)


def shifted_correlation(field, rows, lag, shift):
    """Correlation of the band rows at (y, t) with the field at (y - shift, t + lag) over the whole clip."""
    ya, yb = rows
    before = field[:-lag, ya + shift:yb]
    after = field[lag:, ya:yb - shift]
    a, b = before.ravel(), after.ravel()
    if a.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def field_transport(field, bands, column_width, dt):
    """Per band the upward speed (widths/s) that best aligns the field with its later self over the whole
    clip, and the mean gain of that alignment over no shift."""
    lag = max(1, round(TRANSPORT_LAG_SECONDS / dt))
    max_shift = max(1, round(TRANSPORT_MAX_SHIFT_WIDTHS * column_width * lag * dt))
    speeds, gains = [], []
    for ya, yb in bands:
        shifts = range(0, min(max_shift, (yb - ya) // 2) + 1)
        correlations = [shifted_correlation(field, (ya, yb), lag, shift) for shift in shifts]
        if not np.isfinite(correlations[0]):
            speeds.append(np.nan)
            continue
        best = int(np.nanargmax(correlations))
        gain = correlations[best] - correlations[0]
        speeds.append(float(best / column_width / (lag * dt)) if gain >= TRANSPORT_MIN_GAIN else 0.0)
        gains.append(gain)
    return {"speed": speeds, "gain": gains}


def best_lag(lower, upper, max_lag):
    """Lag (frames >= 0) at which `upper` best follows `lower`, and the correlation there."""
    lower = detrend(lower)
    upper = detrend(upper)
    best = (0, -np.inf)
    for lag in range(0, max_lag + 1):
        a, b = lower[:lower.size - lag], upper[lag:]
        if a.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best[1]:
            best = (lag, corr)
    return best


def centre_phase_lag(centres, bands, dt):
    series = band_centre_series(centres, bands)
    max_lag = max(1, round(PHASE_MAX_LAG_SECONDS / dt))
    lags = []
    for band in range(1, len(bands)):
        lag, corr = best_lag(series[band], series[band - 1], max_lag)
        lags.append(float(lag * dt) if np.isfinite(corr) else np.nan)
    return lags


def detrend(sample):
    sample = np.where(np.isnan(sample), np.nanmean(sample), sample)
    frames = np.arange(sample.size)
    slope, intercept = np.polyfit(frames, sample, 1)
    return sample - (slope * frames + intercept)


def centre_return(centres, bands):
    minima = []
    for sample in band_centre_series(centres, bands):
        sample = detrend(sample)
        if sample.std() < 1e-9:
            minima.append(np.nan)
            continue
        autocorr = [float(np.corrcoef(sample[:-lag], sample[lag:])[0, 1]) for lag in range(1, sample.size // 2)]
        minima.append(float(min(autocorr)) if autocorr else np.nan)
    return minima


def measure_video(rgbs, column_width, dt, silhouette_mask, row_centers):
    masks = [silhouette_mask(rgb) for rgb in rgbs]
    widths, centres = silhouette_profiles(masks, column_width, row_centers)
    details = width_detail(widths, column_width)
    bands = temporal_bands(widths.shape[1])
    shifts = pair_shifts(details, bands, column_width)

    return {
        "moving_pair_fraction": moving_pair_fraction(shifts),
        "width_transport": field_transport(details, bands, column_width, dt),
        "centre_transport": field_transport(centre_detail(centres), bands, column_width, dt),
        "centre_phase_lag": centre_phase_lag(centres, bands, dt),
        "centre_return": centre_return(centres, bands),
    }


def diff_row(label, rv, cv, tolerance, fmt=".2f"):
    diff = cv - rv
    return (label, f"{rv:{fmt}}", f"{cv:{fmt}}", f"diff {diff:+.2f}", bool(np.isfinite(diff) and abs(diff) <= tolerance))


def compare_video(ref, render):
    rows = []
    for band in range(TEMPORAL_BAND_COUNT):
        rows.append(diff_row(f"xviii moving pairs b{band}", ref["moving_pair_fraction"][band],
                             render["moving_pair_fraction"][band], MOVING_PAIR_TOLERANCE))
    for field in ("width", "centre"):
        for band in range(TEMPORAL_BAND_COUNT):
            rows.append(diff_row(f"xviii {field} transport b{band}", ref[f"{field}_transport"]["speed"][band],
                                 render[f"{field}_transport"]["speed"][band], TRANSPORT_SPEED_TOLERANCE))
    for band in range(1, TEMPORAL_BAND_COUNT):
        rows.append(diff_row(f"xviii centre lag b{band}>{band - 1}", ref["centre_phase_lag"][band - 1],
                             render["centre_phase_lag"][band - 1], PHASE_LAG_TOLERANCE))
    for band in range(TEMPORAL_BAND_COUNT):
        rows.append(diff_row(f"xviii centre return b{band}", ref["centre_return"][band], render["centre_return"][band],
                             RETURN_TOLERANCE))
    return rows


def print_video(ref, render):
    print("\nvideo (bands top -> bottom): moving pair fraction, centre return, width / centre transport [width/s]")
    for band in range(TEMPORAL_BAND_COUNT):
        print(f"  band {band}: moving ref {ref['moving_pair_fraction'][band]:.2f} render {render['moving_pair_fraction'][band]:.2f}"
              f"   return ref {ref['centre_return'][band]:+.2f} render {render['centre_return'][band]:+.2f}"
              f"   width ref {ref['width_transport']['speed'][band]:.2f} render {render['width_transport']['speed'][band]:.2f}"
              f"   centre ref {ref['centre_transport']['speed'][band]:.2f} render {render['centre_transport']['speed'][band]:.2f}")
    for field in ("width", "centre"):
        gains = " ".join(f"{r:+.2f}/{c:+.2f}" for r, c in zip(ref[f"{field}_transport"]["gain"], render[f"{field}_transport"]["gain"]))
        print(f"{field} transport gain ref/render by band: {gains}")
    lag = " ".join(f"{r:.2f}/{c:.2f}" for r, c in zip(ref["centre_phase_lag"], render["centre_phase_lag"]))
    print(f"centre phase lag ref/render [s] (b1>b0, b2>b1, b3>b2): {lag}")
