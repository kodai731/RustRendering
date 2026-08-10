pub use thyllore_color_core::{chromaticity_xy, is_saturated, luminance, mccamy_cct, srgb_to_linear};

/// Percentile by linear interpolation over a sorted slice. Returns None on empty input.
pub fn percentile(sorted_values: &[f32], p: f32) -> Option<f32> {
    if sorted_values.is_empty() {
        return None;
    }
    let n = sorted_values.len();
    if n == 1 {
        return Some(sorted_values[0]);
    }
    let rank = p * (n as f32 - 1.0);
    let lo = rank.floor() as usize;
    let frac = rank - lo as f32;
    let hi = lo + 1;
    if hi >= n {
        Some(sorted_values[n - 1])
    } else {
        Some(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))
    }
}

/// Simple threshold mask: true where luminance >= threshold.
pub fn flame_mask(lum: &[f32], _width: usize, _height: usize, threshold: f32) -> Vec<bool> {
    lum.iter().map(|&v| v >= threshold).collect()
}

/// For each row with any true pixel, return (normalized_height_from_bottom h in 0..1,
/// normalized_width = span between first and last true pixel / width).
/// Rows are returned bottom-to-top (row height-1 first, row 0 last).
pub fn row_width_profile(mask: &[bool], width: usize, height: usize) -> Vec<(f32, f32)> {
    let mut profile = Vec::new();
    for row in (0..height).rev() {
        let start = row * width;
        let end = start + width;
        let row_slice = &mask[start..end];
        let first = row_slice.iter().position(|&b| b);
        let last = row_slice.iter().rposition(|&b| b);
        if let (Some(f), Some(l)) = (first, last) {
            let h = (height - 1 - row) as f32 / (height - 1) as f32;
            let w_span = (l - f + 1) as f32 / width as f32;
            profile.push((h, w_span));
        }
    }
    profile
}

/// Per-row mean luminance from bottom to top, normalized so its max is 1.
/// Returns empty if all values are zero.
pub fn vertical_luminance_profile(lum: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut row_means = Vec::with_capacity(height);
    for row in (0..height).rev() {
        let start = row * width;
        let end = start + width;
        let sum: f32 = lum[start..end].iter().sum();
        row_means.push(sum / width as f32);
    }
    let max_val = row_means.iter().cloned().fold(0.0f32, f32::max);
    if max_val <= 1e-9 {
        return Vec::new();
    }
    row_means.iter().map(|&v| v / max_val).collect()
}

/// Crop a vertical profile to the span between the first and last indices where
/// the value is >= threshold. Returns None if no such indices exist.
pub fn crop_profile_to_span(profile: &[f32], threshold: f32) -> Option<Vec<f32>> {
    let first = profile.iter().position(|&v| v >= threshold)?;
    let last = profile.iter().rposition(|&v| v >= threshold)?;
    Some(profile[first..=last].to_vec())
}

/// Least-squares grid search over envelope parameters (peak, base, tail) for a vertical luminance profile.
/// Forward model: m(h) = parametric_height_falloff(h, p, v0, q) * taper(h)
/// where taper(h) = 1.0 + (taper_tip - 1.0) * h^taper_power.
/// Model samples are normalized to max 1, then SSE against the profile.
/// Grid: p in 0.05..=0.80 step 0.05, v0 in 0.0..=0.95 step 0.05, q in 0.5..=4.0 step 0.25.
/// Returns None if profile.len() < 4 or all elements are zero.
pub fn fit_envelope_from_profile(
    profile: &[f32],
    taper_tip: f32,
    taper_power: f32,
) -> Option<(f32, f32, f32)> {
    if profile.len() < 4 || profile.iter().all(|&v| v.abs() < 1e-9) {
        return None;
    }

    let n = profile.len();

    let mut best_sse = f32::INFINITY;
    let mut best_p = 0.05f32;
    let mut best_v0 = 0.0f32;
    let mut best_q = 0.5f32;

    for p_i in 1..=16 {
        let p = p_i as f32 * 0.05;
        for v0_i in 0..=19 {
            let v0 = v0_i as f32 * 0.05;
            for q_i in 2..=16 {
                let q = q_i as f32 * 0.25;

                // Build model samples and normalize to max 1
                let mut model: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let h = i as f64 / (n - 1) as f64;
                    let envelope =
                        thyllore_math_core::parametric_height_falloff(h, p as f64, v0 as f64, q as f64);
                    let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
                    model.push(envelope * taper);
                }
                let model_max = model.iter().cloned().fold(0.0f64, f64::max);
                if model_max < 1e-9 {
                    continue;
                }

                // SSE against profile
                let mut sse = 0.0f32;
                for i in 0..n {
                    let predicted = (model[i] / model_max) as f32;
                    let diff = profile[i] - predicted;
                    sse += diff * diff;
                }

                if sse < best_sse {
                    best_sse = sse;
                    best_p = p;
                    best_v0 = v0;
                    best_q = q;
                }
            }
        }
    }

    Some((best_p, best_v0, best_q))
}

pub fn fit_envelope_from_profile_saturated(
    profile: &[f32],
    taper_tip: f32,
    taper_power: f32,
) -> Option<(f32, f32, f32, f32)> {
    if profile.len() < 4 || profile.iter().all(|&v| v.abs() < 1e-9) {
        return None;
    }

    let n = profile.len();
    let k_values: [f64; 6] = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0];

    let mut best_sse = f32::INFINITY;
    let mut best_p = 0.05f32;
    let mut best_v0 = 0.0f32;
    let mut best_q = 0.5f32;
    let mut best_k = 0.0f32;

    for &k in &k_values {
        for p_i in 1..=16 {
            let p = p_i as f32 * 0.05;
            for v0_i in 0..=19 {
                let v0 = v0_i as f32 * 0.05;
                for q_i in 2..=16 {
                    let q = q_i as f32 * 0.25;

                    // Build model samples and normalize to max 1
                    let mut model: Vec<f64> = Vec::with_capacity(n);
                    for i in 0..n {
                        let h = i as f64 / (n - 1) as f64;
                        let envelope = thyllore_math_core::parametric_height_falloff(
                            h, p as f64, v0 as f64, q as f64,
                        );
                        let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
                        model.push(envelope * taper);
                    }

                    let model_max = model.iter().cloned().fold(0.0f64, f64::max);
                    if model_max < 1e-9 {
                        continue;
                    }

                    // SSE against profile with Beer-Lambert saturation
                    let mut sse = 0.0f32;
                    for i in 0..n {
                        let normalized = (model[i] / model_max) as f32;
                        let predicted = if k > 0.0 {
                            let ek = (-k * normalized as f64).exp();
                            let denom = 1.0 - (-k).exp();
                            ((1.0 - ek) / denom) as f32
                        } else {
                            normalized
                        };
                        let diff = profile[i] - predicted;
                        sse += diff * diff;
                    }

                    if sse < best_sse {
                        best_sse = sse;
                        best_p = p;
                        best_v0 = v0;
                        best_q = q;
                        best_k = k as f32;
                    }
                }
            }
        }
    }

    Some((best_p, best_v0, best_q, best_k))
}

/// Least-squares grid search over tip_ratio and power for the taper portion of a flame profile.
/// Only uses rows with h >= the h of the widest row (the taper applies above the bulge).
/// Grid: tip_ratio in 0.05..=0.6 (step 0.05), power in 0.5..=3.0 (step 0.1).
/// Minimizes sum((w(h)/w_max - (1.0 + (tip - 1.0) * h.powf(power)))^2).
/// Returns None if fewer than 2 taper rows.
pub fn fit_taper(profile: &[(f32, f32)]) -> Option<(f32, f32)> {
    if profile.is_empty() {
        return None;
    }

    // Find the widest row's h value
    let w_max = profile.iter().map(|&(_, w)| w).fold(0.0f32, f32::max);
    let widest_h = profile
        .iter()
        .filter(|&&(_, w)| (w - w_max).abs() < 1e-6)
        .map(|&(h, _)| h)
        .fold(f32::INFINITY, f32::min);

    // Collect taper rows: those with h >= widest_h
    let taper_rows: Vec<(f32, f32)> = profile
        .iter()
        .filter(|&&(h, _)| h >= widest_h - 1e-6)
        .cloned()
        .collect();

    if taper_rows.len() < 2 {
        return None;
    }

    // Grid search
    let mut best_err = f32::INFINITY;
    let mut best_tip = 0.05f32;
    let mut best_power = 0.5f32;

    for tip_i in 1..=12 {
        let tip = tip_i as f32 * 0.05;
        for power_i in 5..=30 {
            let power = power_i as f32 * 0.1;
            let mut err = 0.0f32;
            for &(h, w) in &taper_rows {
                let normalized_w = w / w_max;
                let predicted = 1.0 + (tip - 1.0) * h.powf(power);
                let diff = normalized_w - predicted;
                err += diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_tip = tip;
                best_power = power;
            }
        }
    }

    Some((best_tip, best_power))
}

/// Mean horizontal distance in pixels (normalized by width) between crossings of
/// lo_frac*row_max and hi_frac*row_max scanning from the left edge toward each row's peak,
/// averaged over rows whose max > 0.1.
pub fn edge_width_profile(
    lum: &[f32],
    width: usize,
    height: usize,
    lo_frac: f32,
    hi_frac: f32,
) -> Option<f32> {
    let mut total_distance = 0.0f32;
    let mut count = 0usize;

    for row in 0..height {
        let start = row * width;
        let end = start + width;
        let row_slice = &lum[start..end];

        // Find row max
        let row_max = row_slice.iter().cloned().fold(0.0f32, f32::max);
        if row_max <= 0.1 {
            continue;
        }

        // Find peak position
        let peak_pos = row_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let lo_threshold = lo_frac * row_max;
        let hi_threshold = hi_frac * row_max;

        // Scan from left edge toward peak to find crossings
        let mut lo_crossing: Option<f32> = None;
        let mut hi_crossing: Option<f32> = None;

        for i in 0..=peak_pos {
            let val = row_slice[i];
            if lo_crossing.is_none() && val >= lo_threshold {
                // Linear interpolation between previous and current
                if i == 0 {
                    lo_crossing = Some(0.0);
                } else {
                    let prev = row_slice[i - 1];
                    let frac = (lo_threshold - prev) / (val - prev);
                    lo_crossing = Some((i - 1) as f32 + frac);
                }
            }
            if hi_crossing.is_none() && val >= hi_threshold {
                if i == 0 {
                    hi_crossing = Some(0.0);
                } else {
                    let prev = row_slice[i - 1];
                    let frac = (hi_threshold - prev) / (val - prev);
                    hi_crossing = Some((i - 1) as f32 + frac);
                }
            }
        }

        if let (Some(lo), Some(hi)) = (lo_crossing, hi_crossing) {
            total_distance += (hi - lo).abs();
            count += 1;
        }
    }

    if count == 0 {
        None
    } else {
        Some(total_distance / count as f32 / width as f32)
    }
}

/// On the left boundary x(h) of the mask: detrend by linear fit over h,
/// return (stddev of residual normalized by width, mean run length in rows between
/// sign changes of the residual — the dominant half-wavelength).
/// Returns None if fewer than 3 boundary points.
pub fn boundary_wiggle(mask: &[bool], width: usize, height: usize) -> Option<(f32, f32)> {
    // Extract left boundary: for each row with true pixels, find the first (leftmost) true pixel
    let mut boundary: Vec<(f32, f32)> = Vec::new();
    for row in (0..height).rev() {
        let start = row * width;
        let end = start + width;
        let row_slice = &mask[start..end];
        if let Some(first) = row_slice.iter().position(|&b| b) {
            let h = (height - 1 - row) as f32 / (height - 1) as f32;
            boundary.push((h, first as f32));
        }
    }

    if boundary.len() < 3 {
        return None;
    }

    // Linear fit: x = a + b*h, using least squares
    let n = boundary.len() as f32;
    let sum_h: f32 = boundary.iter().map(|(h, _)| h).sum();
    let sum_x: f32 = boundary.iter().map(|(_, x)| x).sum();
    let sum_hh: f32 = boundary.iter().map(|(h, _)| h * h).sum();
    let sum_hx: f32 = boundary.iter().map(|(h, x)| h * x).sum();

    let denom = n * sum_hh - sum_h * sum_h;
    if denom.abs() < 1e-9 {
        return None;
    }

    let b = (n * sum_hx - sum_h * sum_x) / denom;
    let a = (sum_x - b * sum_h) / n;

    // Compute residuals
    let residuals: Vec<f32> = boundary.iter().map(|(h, x)| x - (a + b * h)).collect();

    // Stddev of residual normalized by width
    let mean_r: f32 = residuals.iter().sum::<f32>() / n;
    let variance: f32 = residuals
        .iter()
        .map(|r| (r - mean_r) * (r - mean_r))
        .sum::<f32>()
        / n;
    let stddev = (variance).sqrt() / width as f32;

    // Mean run length between sign changes of residual
    let signs: Vec<i32> = residuals.iter().map(|r| r.signum() as i32).collect();
    let mut run_lengths: Vec<usize> = Vec::new();
    let mut current_run = 1usize;

    for i in 1..signs.len() {
        if signs[i] != signs[i - 1] && signs[i] != 0 && signs[i - 1] != 0 {
            run_lengths.push(current_run);
            current_run = 1;
        } else {
            current_run += 1;
        }
    }
    run_lengths.push(current_run);

    let mean_run_length: f32 = run_lengths.iter().sum::<usize>() as f32 / run_lengths.len() as f32;

    Some((stddev, mean_run_length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_round_trip_anchor_points() {
        // Black
        assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-6);
        // White
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        // Boundary value 0.04045 -> 0.04045 / 12.92
        let boundary = srgb_to_linear(0.04045);
        assert!((boundary - 0.04045 / 12.92).abs() < 1e-6);
    }

    #[test]
    fn test_luminance_white_is_one() {
        let white: [f32; 3] = [1.0, 1.0, 1.0];
        assert!((luminance(white) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_saturated() {
        assert!(!is_saturated([249, 249, 249]));
        assert!(is_saturated([250, 0, 0]));
        assert!(is_saturated([0, 250, 0]));
        assert!(is_saturated([0, 0, 255]));
    }

    #[test]
    fn test_chromaticity_xy_black_is_none() {
        let black: [f32; 3] = [0.0, 0.0, 0.0];
        assert!(chromaticity_xy(black).is_none());
    }

    #[test]
    fn test_mccamy_cct_d65_white() {
        // D65 white point: x=0.3127, y=0.3290 -> should be close to 6500K
        let xy: [f32; 2] = [0.3127, 0.3290];
        let cct = mccamy_cct(xy);
        assert!(
            cct.is_some(),
            "mccamy_cct returned None for D65 white point"
        );
        let cct = cct.unwrap();
        assert!(
            (cct - 6500.0).abs() < 300.0,
            "D65 CCT {:.1}K is not within 300K of 6500K",
            cct
        );
    }

    #[test]
    fn test_percentile_basics() {
        let values: [f32; 5] = [0.0, 25.0, 50.0, 75.0, 100.0];

        assert!(percentile(&[], 0.5).is_none());
        assert!((percentile(&values, 0.0).unwrap() - 0.0).abs() < 1e-6);
        assert!((percentile(&values, 1.0).unwrap() - 100.0).abs() < 1e-6);
        assert!((percentile(&values, 0.5).unwrap() - 50.0).abs() < 1e-6);
        assert!((percentile(&values, 0.25).unwrap() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_flame_mask_rectangle() {
        // 3x3 image: all bright except bottom-left corner
        let lum = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mask = flame_mask(&lum, 3, 3, 0.5);
        assert_eq!(
            mask,
            [false, true, true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_flame_mask_all_below() {
        let lum = [0.1, 0.2, 0.3];
        let mask = flame_mask(&lum, 3, 1, 0.5);
        assert_eq!(mask, [false, false, false]);
    }

    #[test]
    fn test_row_width_profile_rectangle() {
        // 5x3 mask: a centered rectangle (columns 1-3 true in all rows)
        let mask = [
            false, true, true, true, false, // row 0 (top)
            false, true, true, true, false, // row 1
            false, true, true, true, false, // row 2 (bottom)
        ];
        let profile = row_width_profile(&mask, 5, 3);
        assert_eq!(profile.len(), 3);
        // Bottom row (h=0): span = 3/5 = 0.6
        assert!((profile[0].0 - 0.0).abs() < 1e-6);
        assert!((profile[0].1 - 0.6).abs() < 1e-6);
        // Middle row (h=0.5): span = 3/5 = 0.6
        assert!((profile[1].0 - 0.5).abs() < 1e-6);
        assert!((profile[1].1 - 0.6).abs() < 1e-6);
        // Top row (h=1.0): span = 3/5 = 0.6
        assert!((profile[2].0 - 1.0).abs() < 1e-6);
        assert!((profile[2].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_row_width_profile_triangle() {
        // 5x3 mask: triangle widening toward top (narrow at bottom, wide at top)
        let mask = [
            false, false, true, false, false, // row 0 (top) — wide: just center
            false, true, true, true, false, // row 1 — wider
            true, true, true, true, true, // row 2 (bottom) — widest
        ];
        let profile = row_width_profile(&mask, 5, 3);
        assert_eq!(profile.len(), 3);
        // Bottom row (h=0): span = 5/5 = 1.0
        assert!((profile[0].0 - 0.0).abs() < 1e-6);
        assert!((profile[0].1 - 1.0).abs() < 1e-6);
        // Middle row (h=0.5): span = 3/5 = 0.6
        assert!((profile[1].0 - 0.5).abs() < 1e-6);
        assert!((profile[1].1 - 0.6).abs() < 1e-6);
        // Top row (h=1.0): span = 1/5 = 0.2
        assert!((profile[2].0 - 1.0).abs() < 1e-6);
        assert!((profile[2].1 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_row_width_profile_empty_rows_skipped() {
        // 3x3 mask: only middle row has true pixels
        let mask = [false, false, false, true, true, true, false, false, false];
        let profile = row_width_profile(&mask, 3, 3);
        assert_eq!(profile.len(), 1);
        // Middle row is at h=0.5, span = 3/3 = 1.0
        assert!((profile[0].0 - 0.5).abs() < 1e-6);
        assert!((profile[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_luminance_profile_basics() {
        // 2x3 image: bottom row all 1.0, middle row all 0.5, top row all 0.0
        let lum = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
        let profile = vertical_luminance_profile(&lum, 2, 3);
        assert_eq!(profile.len(), 3);
        // Bottom-to-top: 1.0, 0.5, 0.0 (normalized by max=1.0)
        assert!((profile[0] - 1.0).abs() < 1e-6);
        assert!((profile[1] - 0.5).abs() < 1e-6);
        assert!((profile[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_luminance_profile_all_zero() {
        let lum = [0.0, 0.0, 0.0];
        let profile = vertical_luminance_profile(&lum, 3, 1);
        assert!(profile.is_empty());
    }

    #[test]
    fn test_vertical_luminance_profile_normalization() {
        // 2x2 image: bottom row mean=2.0, top row mean=1.0 -> normalized: 1.0, 0.5
        let lum = [1.0, 1.0, 2.0, 2.0];
        let profile = vertical_luminance_profile(&lum, 2, 2);
        assert_eq!(profile.len(), 2);
        assert!((profile[0] - 1.0).abs() < 1e-6);
        assert!((profile[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fit_taper_triangle_narrowing_to_tip() {
        // Triangle mask: widest at bottom (h=0), narrowing toward top (h=1)
        // Profile from row_width_profile of a triangle:
        // h=0: w=1.0, h=0.25: w=0.75, h=0.5: w=0.5, h=0.75: w=0.25, h=1.0: w=0.0 (single pixel = 0.2)
        // The widest row is at h=0, so taper rows are all rows with h >= 0
        // This means all rows are taper rows
        let profile: [(f32, f32); 5] = [
            (0.0, 1.0),
            (0.25, 0.75),
            (0.5, 0.5),
            (0.75, 0.25),
            (1.0, 0.2),
        ];
        let result = fit_taper(&profile);
        assert!(
            result.is_some(),
            "fit_taper should return Some for triangle profile"
        );
        let (tip, power) = result.unwrap();
        // For a linear taper from 1.0 to ~0.2, tip_ratio should be small and power close to 1.0
        assert!(tip > 0.0 && tip <= 0.6, "tip_ratio {} out of range", tip);
        assert!(power >= 0.5 && power <= 3.0, "power {} out of range", power);
    }

    #[test]
    fn test_fit_taper_empty() {
        let profile: [(f32, f32); 0] = [];
        assert!(fit_taper(&profile).is_none());
    }

    #[test]
    fn test_fit_taper_single_row() {
        let profile = [(0.5, 0.5)];
        assert!(fit_taper(&profile).is_none());
    }

    #[test]
    fn test_edge_width_profile_hard_edge() {
        // 5x3 image: each row has a sharp step from 0 to 1 at column 2
        // Row 0 (bottom): [0, 0, 1, 1, 1] -> peak at col 4, lo=0.1*1=0.1 crosses at col 2, hi=0.9*1=0.9 crosses at col 2
        // So distance between crossings is ~0 for a hard edge
        let lum = [
            0.0, 0.0, 1.0, 1.0, 1.0, // row 0 (bottom)
            0.0, 0.0, 1.0, 1.0, 1.0, // row 1
            0.0, 0.0, 1.0, 1.0, 1.0, // row 2 (top)
        ];
        let result = edge_width_profile(&lum, 5, 3, 0.1, 0.9);
        assert!(
            result.is_some(),
            "edge_width_profile should return Some for hard edge"
        );
        let width = result.unwrap();
        // For a hard edge step [0,0,1,1,1], lo=0.1 crosses at 1+(0.1-0)/(1-0)=1.1,
        // hi=0.9 crosses at 1+(0.9-0)/(1-0)=1.9, distance=0.8, normalized by width=5 -> 0.16
        assert!(
            (width - 0.16).abs() < 0.01,
            "hard edge width {} should be ~0.16",
            width
        );
    }

    #[test]
    fn test_edge_width_profile_gradient() {
        // 5x1 image: gradient [0.0, 0.25, 0.5, 0.75, 1.0]
        // lo_frac=0.1 -> threshold=0.1, crosses between col 0 (0.0) and col 1 (0.25): at 0 + (0.1-0.0)/(0.25-0.0) = 0.4
        // hi_frac=0.9 -> threshold=0.9, crosses between col 3 (0.75) and col 4 (1.0): at 3 + (0.9-0.75)/(1.0-0.75) = 3.6
        // distance = 3.6 - 0.4 = 3.2, normalized by width=5 -> 0.64
        let lum = [0.0, 0.25, 0.5, 0.75, 1.0];
        let result = edge_width_profile(&lum, 5, 1, 0.1, 0.9);
        assert!(result.is_some());
        let width = result.unwrap();
        assert!(
            (width - 0.64).abs() < 0.01,
            "gradient edge width {} should be ~0.64",
            width
        );
    }

    #[test]
    fn test_edge_width_profile_all_below_threshold() {
        let lum = [0.05, 0.05, 0.05];
        let result = edge_width_profile(&lum, 3, 1, 0.1, 0.9);
        assert!(
            result.is_none(),
            "edge_width_profile should return None when all rows below threshold"
        );
    }

    #[test]
    fn test_boundary_wiggle_straight_edge() {
        // 5x5 mask: straight vertical edge at column 2 (all rows have first true pixel at col 2)
        let mask = [
            false, false, true, true, true, // row 0 (top)
            false, false, true, true, true, // row 1
            false, false, true, true, true, // row 2
            false, false, true, true, true, // row 3
            false, false, true, true, true, // row 4 (bottom)
        ];
        let result = boundary_wiggle(&mask, 5, 5);
        assert!(
            result.is_some(),
            "boundary_wiggle should return Some for straight edge"
        );
        let (stddev, _) = result.unwrap();
        // Straight edge has no wiggle, so stddev should be ~0
        assert!(
            stddev < 1e-6,
            "straight edge stddev {} should be near zero",
            stddev
        );
    }

    #[test]
    fn test_boundary_wiggle_wiggled_edge() {
        // 5x5 mask: wiggled edge alternating between columns 1 and 3
        let mask = [
            false, true, true, true, true, // row 0 (top): first at col 1
            false, false, true, true, true, // row 1: first at col 2
            false, true, true, true, true, // row 2: first at col 1
            false, false, true, true, true, // row 3: first at col 2
            false, true, true, true, true, // row 4 (bottom): first at col 1
        ];
        let result = boundary_wiggle(&mask, 5, 5);
        assert!(
            result.is_some(),
            "boundary_wiggle should return Some for wiggled edge"
        );
        let (stddev, mean_run) = result.unwrap();
        // Wiggled edge should have non-zero stddev
        assert!(
            stddev > 0.0,
            "wiggled edge stddev {} should be positive",
            stddev
        );
        // Mean run length should be at least 1
        assert!(
            mean_run >= 1.0,
            "mean run length {} should be >= 1",
            mean_run
        );
    }

    #[test]
    fn test_boundary_wiggle_too_few_points() {
        // Only 2 rows with true pixels -> fewer than 3 boundary points
        let mask = [false, false, false, true, true, true];
        let result = boundary_wiggle(&mask, 3, 2);
        assert!(
            result.is_none(),
            "boundary_wiggle should return None for < 3 boundary points"
        );
    }

    #[test]
    fn test_fit_envelope_from_profile_recovery() {
        // Synthesize a profile from known parameters: p=0.35, v0=0.45, q=1.5, taper_tip=0.10, taper_power=1.4
        let samples = 64;
        let true_p: f64 = 0.35;
        let true_v0: f64 = 0.45;
        let true_q: f64 = 1.5;
        let taper_tip: f32 = 0.10;
        let taper_power: f32 = 1.4;

        let mut model: Vec<f64> = Vec::with_capacity(samples);
        for i in 0..samples {
            let h = i as f64 / (samples - 1) as f64;
            let envelope = thyllore_math_core::parametric_height_falloff(h, true_p, true_v0, true_q);
            let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
            model.push(envelope * taper);
        }

        let model_max = model.iter().cloned().fold(0.0f64, f64::max);
        let profile: Vec<f32> = model.iter().map(|&v| (v / model_max) as f32).collect();

        let result = fit_envelope_from_profile(&profile, taper_tip, taper_power);
        assert!(
            result.is_some(),
            "fit should return Some for synthetic profile"
        );

        let (p, v0, q) = result.unwrap();

        // Assert recovery within one grid step of each true value
        assert!(
            (p - true_p as f32).abs() <= 0.05,
            "peak {} should be within 0.05 of {}",
            p,
            true_p
        );
        assert!(
            (v0 - true_v0 as f32).abs() <= 0.05,
            "base {} should be within 0.05 of {}",
            v0,
            true_v0
        );
        assert!(
            (q - true_q as f32).abs() <= 0.25,
            "tail {} should be within 0.25 of {}",
            q,
            true_q
        );
    }

    #[test]
    fn test_fit_envelope_from_profile_short_input() {
        let profile = [0.0, 0.5, 1.0];
        let result = fit_envelope_from_profile(&profile, 0.1, 1.4);
        assert!(
            result.is_none(),
            "should return None for profile with < 4 elements"
        );
    }

    #[test]
    fn test_fit_envelope_from_profile_all_zero() {
        let profile = [0.0, 0.0, 0.0, 0.0, 0.0];
        let result = fit_envelope_from_profile(&profile, 0.1, 1.4);
        assert!(result.is_none(), "should return None for all-zero profile");
    }

    #[test]
    fn test_crop_profile_to_span() {
        let profile = [0.0, 0.0, 0.5, 1.0, 0.8, 0.3, 0.0, 0.0];
        let cropped = crop_profile_to_span(&profile, 0.05);
        assert_eq!(cropped, Some(vec![0.5, 1.0, 0.8, 0.3]));
    }

    #[test]
    fn test_crop_profile_to_span_all_below() {
        let profile = [0.0, 0.0, 0.0];
        let cropped = crop_profile_to_span(&profile, 0.05);
        assert!(cropped.is_none());
    }

    #[test]
    fn test_crop_profile_to_span_single() {
        let profile = [0.0, 0.5, 0.0];
        let cropped = crop_profile_to_span(&profile, 0.05);
        assert_eq!(cropped, Some(vec![0.5]));
    }

    #[test]
    fn test_fit_envelope_from_profile_saturated_linear_matches() {
        // Synthesize a linear (k=0) profile from known parameters: p=0.25, v0=0.05, q=1.25
        let samples = 64;
        let true_p: f64 = 0.25;
        let true_v0: f64 = 0.05;
        let true_q: f64 = 1.25;
        let taper_tip: f32 = 0.10;
        let taper_power: f32 = 1.4;

        let mut model: Vec<f64> = Vec::with_capacity(samples);
        for i in 0..samples {
            let h = i as f64 / (samples - 1) as f64;
            let envelope = thyllore_math_core::parametric_height_falloff(h, true_p, true_v0, true_q);
            let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
            model.push(envelope * taper);
        }

        let model_max = model.iter().cloned().fold(0.0f64, f64::max);
        let profile: Vec<f32> = model.iter().map(|&v| (v / model_max) as f32).collect();

        // Saturated fit should choose k=0 and match the linear fit
        let sat_result = fit_envelope_from_profile_saturated(&profile, taper_tip, taper_power);
        assert!(sat_result.is_some(), "saturated fit should return Some");
        let (sat_p, sat_v0, sat_q, sat_k) = sat_result.unwrap();

        // Linear fit for comparison
        let lin_result = fit_envelope_from_profile(&profile, taper_tip, taper_power);
        assert!(lin_result.is_some(), "linear fit should return Some");
        let (lin_p, lin_v0, lin_q) = lin_result.unwrap();

        // Saturated version should choose k=0
        assert_eq!(
            sat_k, 0.0,
            "saturated fit should choose k=0 for linear profile"
        );

        // Should return same (peak, base, tail) as linear fit
        assert!(
            (sat_p - lin_p).abs() < 1e-6,
            "peak mismatch: {} vs {}",
            sat_p,
            lin_p
        );
        assert!(
            (sat_v0 - lin_v0).abs() < 1e-6,
            "base mismatch: {} vs {}",
            sat_v0,
            lin_v0
        );
        assert!(
            (sat_q - lin_q).abs() < 1e-6,
            "tail mismatch: {} vs {}",
            sat_q,
            lin_q
        );
    }

    #[test]
    fn test_fit_envelope_from_profile_saturated_better_than_linear() {
        // Synthesize a profile with k=4 saturation from known parameters: p=0.25, v0=0.05, q=1.25
        let samples = 64;
        let true_p: f64 = 0.25;
        let true_v0: f64 = 0.05;
        let true_q: f64 = 1.25;
        let taper_tip: f32 = 0.10;
        let taper_power: f32 = 1.4;
        let k_sat: f64 = 4.0;

        let mut model: Vec<f64> = Vec::with_capacity(samples);
        for i in 0..samples {
            let h = i as f64 / (samples - 1) as f64;
            let envelope = thyllore_math_core::parametric_height_falloff(h, true_p, true_v0, true_q);
            let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
            model.push(envelope * taper);
        }

        let model_max = model.iter().cloned().fold(0.0f64, f64::max);
        // Apply Beer-Lambert saturation to create the profile
        let profile: Vec<f32> = model
            .iter()
            .map(|&v| {
                let normalized = (v / model_max) as f32;
                let ek = (-k_sat * normalized as f64).exp();
                let denom = 1.0 - (-k_sat).exp();
                ((1.0 - ek) / denom) as f32
            })
            .collect();

        // Compute error of linear fit's (peak, base, tail) against true values
        let lin_result = fit_envelope_from_profile(&profile, taper_tip, taper_power);
        assert!(lin_result.is_some(), "linear fit should return Some");
        let (lin_p, lin_v0, lin_q) = lin_result.unwrap();
        let lin_error = (lin_p - true_p as f32).abs()
            + (lin_v0 - true_v0 as f32).abs()
            + (lin_q - true_q as f32).abs();

        // Compute error of saturated fit's (peak, base, tail) against true values
        let sat_result = fit_envelope_from_profile_saturated(&profile, taper_tip, taper_power);
        assert!(sat_result.is_some(), "saturated fit should return Some");
        let (sat_p, sat_v0, sat_q, _) = sat_result.unwrap();
        let sat_error = (sat_p - true_p as f32).abs()
            + (sat_v0 - true_v0 as f32).abs()
            + (sat_q - true_q as f32).abs();

        // Saturated version's error should be <= linear version's error
        assert!(
            sat_error <= lin_error,
            "saturated error ({:.4}) should be <= linear error ({:.4})",
            sat_error,
            lin_error
        );
    }

    #[test]
    fn test_fit_envelope_from_profile_saturated_k_choice() {
        // Synthesize a profile with k=4 saturation from known parameters: p=0.25, v0=0.05, q=1.25
        let samples = 64;
        let true_p: f64 = 0.25;
        let true_v0: f64 = 0.05;
        let true_q: f64 = 1.25;
        let taper_tip: f32 = 0.10;
        let taper_power: f32 = 1.4;
        let k_sat: f64 = 4.0;

        let mut model: Vec<f64> = Vec::with_capacity(samples);
        for i in 0..samples {
            let h = i as f64 / (samples - 1) as f64;
            let envelope = thyllore_math_core::parametric_height_falloff(h, true_p, true_v0, true_q);
            let taper = 1.0 + (taper_tip as f64 - 1.0) * h.powf(taper_power as f64);
            model.push(envelope * taper);
        }

        let model_max = model.iter().cloned().fold(0.0f64, f64::max);
        // Apply Beer-Lambert saturation to create the profile
        let profile: Vec<f32> = model
            .iter()
            .map(|&v| {
                let normalized = (v / model_max) as f32;
                let ek = (-k_sat * normalized as f64).exp();
                let denom = 1.0 - (-k_sat).exp();
                ((1.0 - ek) / denom) as f32
            })
            .collect();

        // Saturated fit should choose k >= 2 for a k=4 synthesized profile
        let sat_result = fit_envelope_from_profile_saturated(&profile, taper_tip, taper_power);
        assert!(sat_result.is_some(), "saturated fit should return Some");
        let (_, _, _, sat_k) = sat_result.unwrap();
        assert!(
            sat_k >= 2.0,
            "saturated fit should choose k >= 2 for k=4 synthesis, got {}",
            sat_k
        );
    }
}
