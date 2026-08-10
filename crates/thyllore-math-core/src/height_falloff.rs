const HEIGHT_FALLOFF_PEAK_MIN: f64 = 0.05;
const HEIGHT_FALLOFF_PEAK_MAX: f64 = 0.8;
const HEIGHT_FALLOFF_BASE_MIN: f64 = 0.0;
const HEIGHT_FALLOFF_BASE_MAX: f64 = 0.95;
const HEIGHT_FALLOFF_TAIL_MIN: f64 = 0.5;
const HEIGHT_FALLOFF_TAIL_MAX: f64 = 4.0;
const HEIGHT_FALLOFF_DENOM_EPSILON: f64 = 1e-9;

/// Parametric height falloff using envelope parameters.
///
/// p = peak.clamp(HEIGHT_FALLOFF_PEAK_MIN, HEIGHT_FALLOFF_PEAK_MAX), v0 = base.clamp(HEIGHT_FALLOFF_BASE_MIN, HEIGHT_FALLOFF_BASE_MAX), q = tail.clamp(HEIGHT_FALLOFF_TAIL_MIN, HEIGHT_FALLOFF_TAIL_MAX)
/// if h < p: v0 + (1.0 - v0) * S(h/p)
/// else: (1.0 - S((h-p)/(1.0-p))).powf(q)
/// Guard: when p >= 1-epsilon and h >= p, return 0 (denominator tiny).
pub fn parametric_height_falloff(h: f64, peak: f64, base: f64, tail: f64) -> f64 {
    let p = peak.clamp(HEIGHT_FALLOFF_PEAK_MIN, HEIGHT_FALLOFF_PEAK_MAX);
    let v0 = base.clamp(HEIGHT_FALLOFF_BASE_MIN, HEIGHT_FALLOFF_BASE_MAX);
    let q = tail.clamp(HEIGHT_FALLOFF_TAIL_MIN, HEIGHT_FALLOFF_TAIL_MAX);

    let result = if h < p {
        v0 + (1.0 - v0) * crate::smooth_step(h / p)
    } else {
        let denom = 1.0 - p;
        if denom < HEIGHT_FALLOFF_DENOM_EPSILON {
            0.0
        } else {
            (1.0 - crate::smooth_step((h - p) / denom)).powf(q)
        }
    };

    result.clamp(0.0, 1.0)
}
